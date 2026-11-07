"""PoC black-box baseline: joint x time CNN (REPRODUCED TRAINING) on CARE-PD, same 5 subject-grouped folds as
poc_symptom.py. Input = pelvis-centred metric 3D joints (no body-size normalisation, as a typical pipeline),
60-frame clips (stride 30), walk = mean of clip probabilities. Explanations: Grad-CAM (last temporal conv,
joint x time) and Integrated Gradients, aggregated over clips into joint-group shares.
Device: GPU only if C:\\Works\\KSEF26\\.gpu_busy is absent, else CPU (BelowNormal, 2 threads).
Run: .venv/Scripts/python.exe scripts/poc_blackbox.py [epochs]"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("XAIPD_THREADS", "2")
from xaipd import memguard as MG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from xaipd import symptoms as SY  # noqa: E402
from xaipd.common import RESULTS, ROOT, write_csv  # noqa: E402

MG.limit_torch()
BUSY = Path(r"C:\Works\KSEF26\.gpu_busy")
def _gpu_free():  # free if no lock, or the lock is ours (written by scripts/gpu_queue.sh)
    try:
        return (not BUSY.exists()) or BUSY.read_text(encoding="utf-8", errors="ignore").startswith("parkinson")
    except OSError:
        return False


DEV = "cuda" if (torch.cuda.is_available() and _gpu_free() and os.environ.get("BB_CPU") != "1") else "cpu"
OUT = RESULTS / "poc"; OUT.mkdir(parents=True, exist_ok=True)
MOUT = ROOT / "models" / "safe" / "poc_blackbox"; MOUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CLIP, STRIDE = 60, 30
GROUPS = {"head+shoulders": [0, 1, 2], "arms": [3, 4, 5, 6], "hips": [7, 8], "legs": [9, 10, 11, 12]}
GROUPS_ASYM = {"head+shoulders": [0, 1, 2], "left_arm": [3, 5], "right_arm": [4, 6], "hips": [7, 8], "legs": [9, 10, 11, 12]}
TARGET = {"reduced_arm_swing": ("arms", GROUPS), "short_stride": ("legs", GROUPS), "arm_asymmetry": ("left_arm", GROUPS_ASYM)}
ARCH = os.environ.get("BB_ARCH", "cnn")      # cnn | stgcn
SEED = int(os.environ.get("BB_SEED", "0"))
TAG = os.environ.get("BB_TAG", "")
t0 = time.time(); print("device", DEV, "arch", ARCH, "seed", SEED, flush=True)

walks, meta = SY.load_cache(); y = meta["y"]; groups = meta["subj"]; N = len(walks)
split = json.load(open(OUT / "splits.json"))
folds = [np.array([i for i in range(N) if groups[i] in set(f["test_subjects"])]) for f in split["folds"]]


def prep(w):  # pelvis-centred metric coordinates; zero frames stay zero
    valid = np.abs(w).sum((1, 2)) > 0
    hip = (w[:, SY.J_LHIP] + w[:, SY.J_RHIP]) / 2
    x = (w - hip[:, None]) * valid[:, None, None]
    return x.astype(np.float32)


def clips(x):
    T = len(x)
    if T < CLIP:
        p = np.zeros((CLIP, 13, 3), np.float32); p[CLIP - T:] = x; return [p]
    return [x[s:s + CLIP] for s in range(0, T - CLIP + 1, STRIDE)]


class JTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o, k, s=(1, 1): nn.Sequential(nn.Conv2d(i, o, k, s, ((k[0] - 1) // 2, 0)), nn.BatchNorm2d(o), nn.ReLU())
        self.t1, self.t2, self.t3 = c(3, 32, (9, 1)), c(32, 64, (9, 1), (2, 1)), c(64, 64, (9, 1))
        self.mix = c(64, 128, (1, 13)); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3)); self.feat = None

    def forward(self, x):  # (N,3,T,13)
        x = self.t3(self.t2(self.t1(x))); self.feat = x
        return self.fc(self.mix(x).mean(dim=(2, 3)))


class STGCN(nn.Module):
    """Black-box ST-GCN (Sabo et al. architecture, 4 blocks, 13-joint COCO graph) on 3D joints, 3 classes."""

    def __init__(self):
        super().__init__()
        from xaipd import sabo as S
        self.enc = S.Encoder(cin=3)
        ck = torch.load(S.checkpoints("detectron")[0], map_location="cpu", weights_only=True)
        self.enc.A.copy_(ck["encoder.A"])  # graph adjacency only (structure), no learned weights reused
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 3))

    @property
    def feat(self):
        return self.enc.feat

    def forward(self, x):  # (N,3,T,13)
        return self.fc(self.enc(x).flatten(1))


def to_clips(W):
    X, G = [], []
    for g, w in enumerate(W):
        for c in clips(prep(w)):
            X.append(c.transpose(2, 0, 1)); G.append(g)
    return torch.from_numpy(np.stack(X)), np.array(G)


def walk_probs(m, W, bs=512):
    X, G = to_clips(W); m.eval()
    with torch.no_grad():
        P = torch.cat([torch.softmax(m(X[i:i + bs].to(DEV)), -1).cpu() for i in range(0, len(X), bs)]).numpy()
    return np.stack([P[G == g].mean(0) for g in range(len(W))])


def joint_attr(m, W, method):
    """Mean joint attribution share per walk for the expected-severity output."""
    X, G = to_clips(W); m.eval(); out = []
    sev = torch.arange(3, dtype=torch.float32, device=DEV)
    for i in range(0, len(X), 256):
        x = X[i:i + 256].to(DEV)
        if method == "gradcam":
            x = x.clone().requires_grad_(True)
            s = (torch.softmax(m(x), -1) * sev).sum()
            g, = torch.autograd.grad(s, m.feat); a = m.feat.detach()
            cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))  # (n,T/2,13)
            out.append(cam.sum(1).cpu().numpy())
        else:  # integrated gradients, zero baseline (all joints at the pelvis)
            tot = torch.zeros_like(x)
            for al in torch.linspace(1 / 16, 1, 16, device=DEV):
                xi = (al * x).requires_grad_(True)
                g, = torch.autograd.grad((torch.softmax(m(xi), -1) * sev).sum(), xi); tot += g
            out.append((x * tot / 16).abs().sum(dim=(1, 2)).detach().cpu().numpy())
    A = np.concatenate(out)
    A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    return np.stack([A[G == g].mean(0) for g in range(len(W))])  # (walks,13)


P = np.zeros((N, 3)); inj_rows, rob_rows = [], []
for f, te in enumerate(folds):
    tf = time.time(); torch.manual_seed(1000 * SEED + f)
    tr = np.setdiff1d(np.arange(N), te)
    Xtr, Gtr = to_clips([walks[i] for i in tr]); Ytr = torch.from_numpy(y[tr][Gtr]).long()
    m = (STGCN() if ARCH == 'stgcn' else JTCNN()).to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    cnt = torch.bincount(Ytr, minlength=3).float(); lossf = nn.CrossEntropyLoss(weight=((cnt.sum() / cnt) / 3).to(DEV))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); lossf(m(Xtr[b].to(DEV)), Ytr[b].to(DEV)).backward(); opt.step()
        print(f"fold{f} ep{ep} {time.time()-tf:.0f}s", flush=True)
    m.eval(); save_file({k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}, str(MOUT / f"fold{f}{TAG}.safetensors"))
    P[te] = walk_probs(m, [walks[i] for i in te])
    # injection
    healthy = [i for i in te if y[i] == 0]; W0 = [walks[i] for i in healthy]
    p0 = walk_probs(m, W0); A0 = {mt: joint_attr(m, W0, mt) for mt in ("gradcam", "ig")}
    np.savez_compressed(OUT / f"bb_attr_healthy_fold{f}{TAG}.npz", idx=np.array(healthy), **A0)
    for kind in ("reduced_arm_swing", "short_stride", "arm_asymmetry"):
        Wk = [SY.inject(w, kind) for w in W0]; p1 = walk_probs(m, Wk)
        tgt, grp = TARGET[kind]
        for mt in ("gradcam", "ig"):
            A1 = joint_attr(m, Wk, mt); dA = A1 - A0[mt]
            gsh = np.stack([dA[:, j].sum(1) for j in grp.values()], 1); names = list(grp)
            for j in range(len(healthy)):
                inj_rows.append(dict(fold=f, model=f"blackbox{TAG}_{mt}", injection=kind, d_expected_severity=float((p1[j] - p0[j]) @ np.arange(3)),
                                     class_increased=int(p1[j].argmax() > p0[j].argmax()), top_group=names[int(np.argmax(gsh[j]))],
                                     hit=int(names[int(np.argmax(gsh[j]))] == tgt), chance=1 / len(grp)))
    # robustness
    sub = list(te[:200]); Ws = [walks[i] for i in sub]; b0 = walk_probs(m, Ws).argmax(1)
    for tag, Wt in (("pad60", [np.concatenate([np.zeros((60, 13, 3), np.float32), w]) for w in Ws]),
                    ("scale0.8", [w * 0.8 for w in Ws]), ("scale1.25", [w * 1.25 for w in Ws])):
        rob_rows.append(dict(fold=f, model="blackbox", test=tag, class_change_rate=float((walk_probs(m, Wt).argmax(1) != b0).mean())))
    print(f"fold {f} done {time.time()-tf:.0f}s total {time.time()-t0:.0f}s", flush=True)

from sklearn.metrics import cohen_kappa_score, f1_score  # noqa: E402
yp = P.argmax(1)
rng = np.random.default_rng(0); boots = {"acc": [], "macro_f1": [], "qwk": []}
for _ in range(1000):
    i = rng.integers(0, N, N)
    boots["acc"].append((yp[i] == y[i]).mean()); boots["macro_f1"].append(f1_score(y[i], yp[i], average="macro"))
    boots["qwk"].append(cohen_kappa_score(y[i], yp[i], weights="quadratic"))
summary = dict(device=DEV, arch=ARCH, seed=SEED, tag=TAG, epochs=EPOCHS, metrics=dict(acc=float((yp == y).mean()), macro_f1=float(f1_score(y, yp, average="macro")),
                                                     qwk=float(cohen_kappa_score(y, yp, weights="quadratic")),
                                                     mae_expected=float(np.abs(P @ np.arange(3) - y).mean())),
               ci95={k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in boots.items()},
               per_cohort_macro_f1={d: float(f1_score(y[meta["ds"] == d], yp[meta["ds"] == d], average="macro")) for d in np.unique(meta["ds"])})
summary["injection"] = {}
for mt in ("gradcam", "ig"):
    for kind in TARGET:
        rr = [r for r in inj_rows if r["model"] == f"blackbox{TAG}_{mt}" and r["injection"] == kind]
        summary["injection"][f"blackbox_{mt}|{kind}"] = dict(n=len(rr), mean_d_expected_severity=float(np.mean([r["d_expected_severity"] for r in rr])),
                                                            class_increase_rate=float(np.mean([r["class_increased"] for r in rr])),
                                                            explanation_hit_rate=float(np.mean([r["hit"] for r in rr])), chance_hit=rr[0]["chance"])
summary["robustness"] = {f"blackbox|{t}": float(np.mean([r["class_change_rate"] for r in rob_rows if r["test"] == t])) for t in ("pad60", "scale0.8", "scale1.25")}
summary["runtime_s"] = round(time.time() - t0, 1); summary["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(summary, open(OUT / f"blackbox_summary{TAG}.json", "w"), indent=1)
write_csv(OUT / f"blackbox_injection_per_walk{TAG}.csv", inj_rows)
np.save(OUT / f"blackbox_test_probs{TAG}.npy", P)
print(json.dumps(summary, indent=1)); MG.report(" poc_blackbox")
