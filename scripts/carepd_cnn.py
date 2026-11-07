"""Gait-motion modality, auxiliary target: REPRODUCED-TRAINING joint x time CNN on CARE-PD
(not public weights). CARE-PD protocol: per-dataset 6-fold participant splits (folds/*_6fold_participants),
UPDRS-gait, macro-F1. Deviation: 3 classes {0,1,>=2} (score 3 is rare), clip = 60 frames (2 s) stride 30,
walk prediction = mean clip softmax. 3 seeds. CPU only, 4 threads, 6 GB guard.
Then XAI on held-out walks (fold-1 models): Grad-CAM / IG / occlusion, seed agreement, E5, E6.
Run: .venv/Scripts/python.exe scripts/carepd_cnn.py [epochs]"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from xaipd import care_pd as CP  # noqa: E402
from xaipd.common import RESULTS, ROOT, spearman, write_csv  # noqa: E402
from xaipd.safe_pickle import safe_load  # noqa: E402

MG.limit_torch()
BUSY = Path(r"C:\Works\KSEF26\.gpu_busy")
while BUSY.exists():  # GenerativeUI is measuring GPU latency -> wait
    print("waiting: .gpu_busy present", flush=True); time.sleep(30)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, flush=True)
OUT = RESULTS / "gait_motion"; OUT.mkdir(parents=True, exist_ok=True)
MOUT = ROOT / "models" / "safe" / "carepd_jtcnn_repro"; MOUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
CLIP, STRIDE, NCLS = 60, 30, 3
ROI_J = [3, 4, 5, 6, 9, 10, 11, 12]
t0 = time.time()


def norm_walk(X3):
    mh = (X3[:, 7] + X3[:, 8]) / 2
    ms = (X3[:, 1] + X3[:, 2]) / 2
    s = np.median(np.linalg.norm(ms - mh, axis=-1))
    return ((X3 - mh[:, None]) / s).astype(np.float32)  # (T,13,3) pelvis-centred, torso-scaled


def clips(x):
    T = x.shape[0]
    if T < CLIP:
        p = np.zeros((CLIP, 13, 3), np.float32); p[CLIP - T:] = x; return [p]
    return [x[s:s + CLIP] for s in range(0, T - CLIP + 1, STRIDE)]


class JTCNN(nn.Module):
    """Joint x time CNN: temporal convs per joint (shared), then a joint-mixing conv. Input (N,3,T,13)."""

    def __init__(self):
        super().__init__()
        c = lambda i, o, k, s=(1, 1): nn.Sequential(nn.Conv2d(i, o, k, s, ((k[0] - 1) // 2, 0)), nn.BatchNorm2d(o), nn.ReLU())
        self.t1, self.t2, self.t3 = c(3, 32, (9, 1)), c(32, 64, (9, 1), (2, 1)), c(64, 64, (9, 1))
        self.mix = c(64, 128, (1, 13))
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, NCLS))
        self.feat = None

    def forward(self, x):
        x = self.t3(self.t2(self.t1(x)))
        self.feat = x  # (N,64,T/2,13) Grad-CAM target
        x = self.mix(x).mean(dim=(2, 3))
        return self.fc(x)


# ---------------------------------------------------------------- data
J, nose = CP.smpl_rest()
data = {}
for ds in CP.UPDRS_SETS:
    W = []
    for w in CP.iter_walks(ds):
        if w["updrs"] is None:
            continue
        X3 = CP.fk_coco13(w["pose"], w["trans"], J, nose)
        W.append(dict(subj=w["subj"].split(":")[1], walk=w["walk"], y=min(int(w["updrs"]), 2), x=norm_walk(X3)))
    folds = safe_load(CP.CDIR / f"{ds}_6fold_participants.pkl")
    data[ds] = (W, folds)
    print(ds, len(W), "walks", {c: sum(w['y'] == c for w in W) for c in range(3)}, f"{time.time()-t0:.0f}s", flush=True)


def to_tensor(ws):
    X, Y, G = [], [], []
    for gi, w in enumerate(ws):
        for c in clips(w["x"]):
            X.append(c.transpose(2, 0, 1)); Y.append(w["y"]); G.append(gi)
    return torch.from_numpy(np.stack(X)), torch.tensor(Y), np.array(G)


def macro_f1(y, p):
    f = []
    for c in range(NCLS):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if (y == c).sum() == 0 and (p == c).sum() == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def train_eval(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, _ = to_tensor(tr); Xte, _, Gte = to_tensor(te)
    m = JTCNN().to(DEV); Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-3)
    cnt = torch.bincount(Ytr, minlength=NCLS).float().clamp(min=1)
    lossf = nn.CrossEntropyLoss(weight=((cnt.sum() / cnt) / NCLS).to(DEV))
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb = Xtr[b] * (1 + 0.05 * torch.randn(len(b), 1, 1, 1, device=DEV))  # light scale aug
            opt.zero_grad(); lossf(m(xb), Ytr[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        P = torch.softmax(m(Xte.to(DEV)), -1).cpu().numpy()
    m = m.cpu()
    pw = np.array([P[Gte == g].mean(0).argmax() for g in range(len(te))])
    return m, pw


summary, per_walk = {}, []
models_fold1 = {}
for ds, (W, folds) in data.items():
    res = {}
    for seed in (0, 1, 2):
        ys, ps = [], []
        for k, f in folds.items():
            tr = [w for w in W if w["subj"] in set(map(str, f["train"]))]
            te = [w for w in W if w["subj"] in set(map(str, f["eval"]))]
            if not te or not tr:
                continue
            ts = time.time()
            m, pw = train_eval(tr, te, seed)
            ys += [w["y"] for w in te]; ps += list(pw)
            per_walk += [dict(dataset=ds, seed=seed, fold=k, walk=w["walk"], y=w["y"], pred=int(p)) for w, p in zip(te, pw)]
            if k == 1:
                models_fold1[(ds, seed)] = (m, te)
                from safetensors.torch import save_file
                save_file(m.state_dict(), str(MOUT / f"{ds}_fold1_seed{seed}.safetensors"))
        ys, ps = np.array(ys), np.array(ps)
        res[f"seed{seed}"] = dict(macro_f1=macro_f1(ys, ps), acc=float((ys == ps).mean()), n=len(ys))
        print(ds, seed, res[f"seed{seed}"], f"{time.time()-t0:.0f}s", flush=True)
    res["macro_f1_mean"] = float(np.mean([res[f"seed{s}"]["macro_f1"] for s in (0, 1, 2)]))
    summary[ds] = res
write_csv(OUT / "carepd_cnn_per_walk.csv", per_walk)
summary["train_runtime_s"] = round(time.time() - t0, 1)
json.dump(summary, open(OUT / "carepd_cnn_e0.json", "w"), indent=1)


# ---------------------------------------------------------------- XAI (largest dataset with fold-1 models)
def gradcam(m, x, cls):
    x = x.clone().requires_grad_(True)
    out = m(x); s = out.gather(1, cls[:, None]).sum()
    g, = torch.autograd.grad(s, m.feat); a = m.feat.detach()
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    return torch.nn.functional.interpolate(cam, size=(CLIP, 13), mode="bilinear", align_corners=False)[:, 0].numpy()


def ig(m, x, cls, steps=32):
    tot = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (a * x).requires_grad_(True)
        g, = torch.autograd.grad(m(xi).gather(1, cls[:, None]).sum(), xi); tot += g
    return (x * tot / steps).abs().sum(1).detach().numpy()


def occlusion(m, x, cls, win=10):
    out = np.zeros((x.shape[0], CLIP, 13), np.float32)
    with torch.no_grad():
        s0 = torch.softmax(m(x), -1).gather(1, cls[:, None])[:, 0]
        for j in range(13):
            for t in range(0, CLIP, win):
                xo = x.clone(); xo[:, :, t:t + win, j] = 0
                out[:, t:t + win, j] = (s0 - torch.softmax(m(xo), -1).gather(1, cls[:, None])[:, 0]).clamp(min=0).numpy()[:, None]
    return out


def metrics(a):
    a = np.maximum(a, 0); tot = a.sum()
    if tot <= 0:
        return None
    js = a.sum(0) / tot
    edge = np.zeros(CLIP, bool); edge[:10] = True; edge[-10:] = True
    return dict(roi_enrichment=float(js[ROI_J].sum() / (len(ROI_J) / 13)), edge_enrichment=float(a[edge].sum() / tot / edge.mean()),
                top_joint_share=float(js.max()), nose_share=float(js[0]),
                pointing_hit=float(np.unravel_index(np.argmax(a), a.shape)[1] in ROI_J))


ds_x = max(data, key=lambda d: len(data[d][0]))
xai = {"dataset": ds_x}
m0, te = models_fold1[(ds_x, 0)]
X = torch.from_numpy(np.stack([clips(w["x"])[len(clips(w["x"])) // 2].transpose(2, 0, 1) for w in te]))
with torch.no_grad():
    cls = m0(X).argmax(1)
A = {}
for s in (0, 1, 2):
    A[("gradcam", s)] = gradcam(models_fold1[(ds_x, s)][0], X, cls)
A[("ig", 0)] = ig(m0, X, cls); A[("occlusion", 0)] = occlusion(m0, X, cls)
rng = np.random.default_rng(0)
for meth in ("gradcam", "ig", "occlusion"):
    rows = [r for r in (metrics(A[(meth, 0)][i]) for i in range(len(X))) if r]
    xai[meth] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
rnd = [metrics(rng.random((CLIP, 13))) for _ in range(len(X))]
xai["baseline_random"] = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
o = np.zeros((CLIP, 13)); o[:, ROI_J] = 1
xai["baseline_oracle"] = metrics(o)
xai["seed_agreement_gradcam"] = float(np.nanmean([np.nanmean([spearman(A[("gradcam", a)][i], A[("gradcam", b)][i]) for i in range(len(X))])
                                                   for a, b in combinations((0, 1, 2), 2)]))
xai["method_agreement"] = {f"{a}-vs-{b}": float(np.nanmean([spearman(A[(a, 0)][i], A[(b, 0)][i]) for i in range(len(X))]))
                           for a, b in (("gradcam", "ig"), ("gradcam", "occlusion"), ("ig", "occlusion"))}
mr = copy.deepcopy(m0); g = torch.Generator().manual_seed(0); e5 = {}
for name, mods in (("head", [mr.fc, mr.mix]), ("head+t3", [mr.t3]), ("head+t2-3", [mr.t2]), ("all", [mr])):
    with torch.no_grad():
        for mod in mods:
            for p_ in mod.parameters():
                p_.copy_(p_.flatten()[torch.randperm(p_.numel(), generator=g)].view_as(p_))
    gc, igr = gradcam(mr, X, cls), ig(mr, X, cls)
    e5[name] = dict(gradcam=float(np.nanmean([abs(spearman(A[("gradcam", 0)][i], gc[i])) for i in range(len(X))])),
                    ig=float(np.nanmean([abs(spearman(A[("ig", 0)][i], igr[i])) for i in range(len(X))])))
xai["E5_abs_spearman_vs_orig"] = e5
steps = np.linspace(0, 1, 11); e6 = {}
for meth in ("gradcam", "ig"):
    for order in ("attr", "rand"):
        dl, il = [], []
        for s in steps:
            k = int(round(s * CLIP * 13)); xd, xi = X.clone(), torch.zeros_like(X)
            for i in range(len(X)):
                r = np.argsort(-A[(meth, 0)][i].ravel()) if order == "attr" else rng.permutation(CLIP * 13)
                tt, jj = np.unravel_index(r[:k], (CLIP, 13))
                xd[i, :, tt, jj] = 0; xi[i, :, tt, jj] = X[i, :, tt, jj]
            with torch.no_grad():
                dl.append(torch.softmax(m0(xd), -1).gather(1, cls[:, None])[:, 0].numpy())
                il.append(torch.softmax(m0(xi), -1).gather(1, cls[:, None])[:, 0].numpy())
        e6[f"{meth}|{order}"] = dict(deletion_auc=float(np.trapz(np.stack(dl, 1), steps, axis=1).mean()),
                                     insertion_auc=float(np.trapz(np.stack(il, 1), steps, axis=1).mean()))
xai["E6"] = e6; xai["n"] = len(X)
summary["xai"] = xai
summary["runtime_s"] = round(time.time() - t0, 1); summary["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(summary, open(OUT / "carepd_cnn_e0.json", "w"), indent=1)
np.savez_compressed(OUT / "attr_carepd_cnn.npz", X=X.numpy(), **{f"{k[0]}_s{k[1]}": v for k, v in A.items()})
print(json.dumps(summary, indent=1))
MG.report(" carepd_cnn")
