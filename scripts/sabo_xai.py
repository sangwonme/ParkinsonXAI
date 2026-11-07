"""Gait-motion modality, main target: Sabo et al. ST-GCN public checkpoints (10 folds, Detectron 2D).
A) E0 on the 15 public labelled sample walks (detectron & kinect2d models on their own input type)
B) External validation on CARE-PD (SMPL -> COCO-13 -> 2D, 4 views, 3 torso scales)
C) XAI: Grad-CAM (last ST-GCN block), Integrated Gradients, occlusion (joint x 10-frame cell);
   ROI/shortcut metrics, fold consistency, method agreement, E5 cascading randomisation, E6 del/ins.
CPU only, memory guard 6 GB, 4 threads. Run: .venv/Scripts/python.exe scripts/sabo_xai.py [n_care_xai]"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xaipd import care_pd as CP  # noqa: E402
from xaipd import sabo as S  # noqa: E402
from xaipd.common import RESULTS, spearman, write_csv  # noqa: E402

MG.limit_torch()
OUT = RESULTS / "gait_motion"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
N_CARE_XAI = int(sys.argv[1]) if len(sys.argv) > 1 else 120
t0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def macro_f1(y, p, classes=(0, 1, 2)):
    f = []
    for c in classes:
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if (y == c).sum() == 0 and (p == c).sum() == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else float("nan")


models = {k: [S.load(p) for p in S.checkpoints(k)] for k in ("detectron", "kinect2d")}
log("loaded", {k: len(v) for k, v in models.items()})

# ---------------------------------------------------------------- A) public samples
samples = {"detectron": sorted((S.DATA / "tri_detectron").rglob("*.csv")),
           "kinect2d": sorted((S.DATA / "kinect_skel_data_2d").rglob("*.csv"))}
e0_rows, sample_inputs = [], []
for kind, files in samples.items():
    for f in files:
        d = S.load_csv(f)
        raws = [S.predict_walk(m, d["x"])[0] for m in models[kind]]
        ens = float(np.mean(raws))
        e0_rows.append(dict(kind=kind, walk=d["name"], T=d["T"], updrs=d["updrs"],
                            pred_ens=int(np.clip(round(ens), 0, 2)), raw_ens=round(ens, 3),
                            fold_acc=float(np.mean([int(np.clip(round(r), 0, 2)) == d["updrs"] for r in raws])),
                            raw_fold_min=round(min(raws), 3), raw_fold_max=round(max(raws), 3)))
        sample_inputs.append((kind, d))
write_csv(OUT / "e0_public_samples.csv", e0_rows)
e0 = {k: dict(n=sum(r["kind"] == k for r in e0_rows),
              acc_ensemble=float(np.mean([r["pred_ens"] == r["updrs"] for r in e0_rows if r["kind"] == k])),
              acc_mean_over_folds=float(np.mean([r["fold_acc"] for r in e0_rows if r["kind"] == k])),
              label_counts={str(u): sum(r["updrs"] == u for r in e0_rows if r["kind"] == k) for u in (0, 1, 2)})
      for k in samples}
log("E0 samples", e0)

# ---------------------------------------------------------------- B) CARE-PD external validation
J, nose = CP.smpl_rest()
walks = []
for ds in CP.UPDRS_SETS:
    for w in CP.iter_walks(ds):
        if w["updrs"] is None:
            continue
        X3 = CP.fk_coco13(w["pose"], w["trans"], J, nose)
        if w["fps"] and abs(w["fps"] - 30) > 1:  # resample to 30 fps
            T = X3.shape[0]; tn = np.arange(0, T, w["fps"] / 30.0)
            X3 = np.stack([np.stack([np.interp(tn, np.arange(T), X3[:, j, c]) for c in range(3)], -1) for j in range(13)], 1)
        walks.append(dict(dataset=ds, subj=w["subj"], walk=w["walk"], updrs=int(w["updrs"]), X3=X3.astype(np.float32)))
log("CARE-PD walks", len(walks), {ds: sum(w["dataset"] == ds for w in walks) for ds in CP.UPDRS_SETS}, f"{time.time()-t0:.0f}s")

ext_rows = []
settings = [(v, 190.0) for v in CP.VIEWS] + [("frontal", 150.0), ("frontal", 230.0)]
MAXW = 8  # up to 8 evenly spaced 120-frame windows per walk (speed; documented deviation)
CACHE = OUT / "e0_carepd_external_per_walk.csv"
if os.environ.get("REUSE_EXT") and CACHE.exists():  # external validation already computed (43 min on CPU)
    import pandas as pd
    ext_rows = pd.read_csv(CACHE).to_dict("records")
    settings_todo = []
    log("reusing", CACHE.name, len(ext_rows))
else:
    settings_todo = settings
for view, torso in settings_todo:
    W, G = [], []
    for gi, w in enumerate(walks):
        ws, _ = S.windows(S.preprocess_xyconf(CP.to_sabo_frame(CP.VIEWS[view](w["X3"]), torso)))
        if len(ws) > MAXW:
            ws = [ws[i] for i in np.linspace(0, len(ws) - 1, MAXW).round().astype(int)]
        W += ws; G += [gi] * len(ws)
    Wt = torch.from_numpy(np.stack(W)); G = np.array(G); del W
    raw_w = np.zeros(len(Wt))
    with torch.no_grad():
        for m in models["detectron"]:
            raw_w += np.concatenate([m(Wt[i:i + 512]).numpy() for i in range(0, len(Wt), 512)])
    raw_w /= len(models["detectron"])
    for gi, w in enumerate(walks):
        raw = float(raw_w[G == gi].mean())
        ext_rows.append(dict(view=view, torso=torso, dataset=w["dataset"], subj=w["subj"], walk=w["walk"],
                             updrs=w["updrs"], raw=raw, pred=int(np.clip(round(raw), 0, 2))))
    del Wt
    log("ext", view, torso, f"{time.time()-t0:.0f}s")
write_csv(OUT / "e0_carepd_external_per_walk.csv", ext_rows)
ext = {}
for view, torso in settings:
    for ds in CP.UPDRS_SETS + ["ALL"]:
        rr = [r for r in ext_rows if r["view"] == view and r["torso"] == torso and (ds == "ALL" or r["dataset"] == ds)]
        y = np.clip(np.array([r["updrs"] for r in rr]), 0, 2); p = np.array([r["pred"] for r in rr]); raw = np.array([r["raw"] for r in rr])
        ext[f"{view}|{int(torso)}|{ds}"] = dict(n=len(rr), acc=float((y == p).mean()), macro_f1=macro_f1(y, p),
                                                mae_raw=float(np.abs(raw - y).mean()), spearman_raw=float(spearman(raw, y)),
                                                pred_dist={str(c): int((p == c).sum()) for c in (0, 1, 2)},
                                                true_dist={str(c): int((y == c).sum()) for c in (0, 1, 2)})
json.dump(ext, open(OUT / "e0_carepd_external.json", "w"), indent=1)
log("external ALL:", {k: (round(v["acc"], 3), round(v["macro_f1"], 3), round(v["spearman_raw"], 3)) for k, v in ext.items() if k.endswith("ALL")})

# ---------------------------------------------------------------- C) XAI
def baseline_like(x):  # all joints at mid-hip (100,100) in the normalised frame
    b = torch.zeros_like(x); b[:, 0] = 100 / S.RES[0] - 0.5; b[:, 1] = 100 / S.RES[1] - 0.5
    return b


def gradcam(m, x):
    x = x.clone().requires_grad_(True)
    out = m(x); g, = torch.autograd.grad(out.sum(), m.encoder.feat)
    a = m.encoder.feat.detach(); w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1, keepdim=True))  # (N,1,60,13)
    cam = torch.nn.functional.interpolate(cam, size=(S.T_IN, 13), mode="bilinear", align_corners=False)
    return cam[:, 0].numpy()  # (N,120,13)


def ig(m, x, steps=32):
    b = baseline_like(x); tot = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (b + a * (x - b)).requires_grad_(True)
        g, = torch.autograd.grad(m(xi).sum(), xi); tot += g
    return ((x - b) * tot / steps).abs().sum(1).detach().numpy()  # (N,120,13)


def occlusion(m, x, win=10):
    b = baseline_like(x); out = np.zeros((x.shape[0], S.T_IN, 13), np.float32)
    with torch.no_grad():
        s0 = m(x)
        for j in range(13):
            for t in range(0, S.T_IN, win):
                xo = x.clone(); xo[:, :, t:t + win, j] = b[:, :, t:t + win, j]
                out[:, t:t + win, j] = (s0 - m(xo)).abs().numpy()[:, None]
    return out


GROUPS = {"head(nose)": [0], "shoulders": [1, 2], "arms(el,wr)": [3, 4, 5, 6], "hips": [7, 8], "legs(kn,ank)": [9, 10, 11, 12]}
ROI_J = [3, 4, 5, 6, 9, 10, 11, 12]  # distal limbs: stride/foot clearance + arm swing (UPDRS 3.10 items)


def metrics(a, pad):
    """a (120,13) >=0 attribution; pad = # zero-padded leading frames."""
    a = np.maximum(a, 0); tot = a.sum()
    if tot <= 0:
        return None
    valid = np.zeros(S.T_IN, bool); valid[pad:] = True
    roi = np.zeros((S.T_IN, 13), bool); roi[np.ix_(valid, ROI_J)] = True
    vt = valid.sum()
    edge = np.zeros(S.T_IN, bool); idx = np.where(valid)[0]; edge[idx[:10]] = True; edge[idx[-10:]] = True
    js = a.sum(0) / tot
    r = dict(pad_frac=pad / S.T_IN, pad_share=float(a[~valid].sum() / tot),
             roi_enrichment=float((a[roi].sum() / a[valid].sum()) / (len(ROI_J) / 13)) if a[valid].sum() > 0 else np.nan,
             edge_enrichment=float((a[edge].sum() / a[valid].sum()) / (edge.sum() / vt)) if a[valid].sum() > 0 else np.nan,
             top_joint_share=float(js.max()), nose_share=float(js[0]),
             pointing_hit=float(roi.ravel()[np.argmax(a)]))
    for g, ids in GROUPS.items():
        r[f"share_{g}"] = float(js[ids].sum())
    return r


def window_of(x):
    ws, pad = S.windows(x)
    return ws[len(ws) // 2], pad  # middle window (deterministic)


xai_sets = {}
xs, pads, tags = [], [], []
for kind, d in sample_inputs:
    if kind != "detectron":
        continue
    w, pad = window_of(d["x"]); xs.append(w); pads.append(pad); tags.append(f"sample:{d['name']}")
xai_sets["public_samples(detectron)"] = (np.stack(xs), np.array(pads), tags)
sel = RNG.choice(len(walks), min(N_CARE_XAI, len(walks)), replace=False)
xs, pads, tags = [], [], []
for i in sel:
    w = walks[i]
    x = S.preprocess_xyconf(CP.to_sabo_frame(CP.VIEWS["frontal"](w["X3"]), 190.0))
    win, pad = window_of(x); xs.append(win); pads.append(pad); tags.append(f"{w['dataset']}:{w['walk']}:{w['updrs']}")
xai_sets["carepd_frontal"] = (np.stack(xs), np.array(pads), tags)

summary = dict(E0_public_samples=e0, runtime={})
fold_models = models["detectron"]
for set_name, (X, P, tags) in xai_sets.items():
    x = torch.from_numpy(X)
    A = {}
    for f, m in enumerate(fold_models):
        A[("gradcam", f)] = gradcam(m, x)
    m0 = fold_models[0]
    A[("ig", 0)] = ig(m0, x); A[("occlusion", 0)] = occlusion(m0, x)
    res = {}
    for meth in ("gradcam", "ig", "occlusion"):
        rows = [metrics(A[(meth, 0)][i], int(P[i])) for i in range(len(X))]
        rows = [r for r in rows if r]
        res[meth] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    # baselines: random map & oracle ROI map
    rnd = [metrics(RNG.random((S.T_IN, 13)), int(p)) for p in P]
    orc = []
    for p in P:
        o = np.zeros((S.T_IN, 13)); o[int(p):, ROI_J] = 1; orc.append(metrics(o, int(p)))
    res["baseline_random"] = {k: float(np.nanmean([r[k] for r in rnd])) for k in rnd[0]}
    res["baseline_oracle"] = {k: float(np.nanmean([r[k] for r in orc])) for k in orc[0]}
    # fold consistency (Grad-CAM) and method agreement (fold 0)
    res["fold_agreement_gradcam_spearman"] = float(np.nanmean([np.nanmean([spearman(A[("gradcam", a)][i], A[("gradcam", b)][i]) for i in range(len(X))])
                                                               for a, b in combinations(range(len(fold_models)), 2)]))
    res["method_agreement"] = {f"{a}-vs-{b}": float(np.nanmean([spearman(A[(a, 0)][i], A[(b, 0)][i]) for i in range(len(X))]))
                               for a, b in (("gradcam", "ig"), ("gradcam", "occlusion"), ("ig", "occlusion"))}
    # E5 cascading randomisation (fold 0 copy)
    import copy
    e5 = {}
    mr = copy.deepcopy(m0); g = torch.Generator().manual_seed(0)
    stages = [("head", [mr.head]), ("head+block4", [mr.encoder.st_gcn_networks[3]]),
              ("head+block3-4", [mr.encoder.st_gcn_networks[2]]), ("all", [mr])]
    for name, mods in stages:  # cumulative
        with torch.no_grad():
            for mod in mods:
                for p_ in mod.parameters():
                    p_.copy_(p_.flatten()[torch.randperm(p_.numel(), generator=g)].view_as(p_))
        gc = gradcam(mr, x); igr = ig(mr, x)
        e5[name] = dict(gradcam=float(np.nanmean([abs(spearman(A[("gradcam", 0)][i], gc[i])) for i in range(len(X))])),
                        ig=float(np.nanmean([abs(spearman(A[("ig", 0)][i], igr[i])) for i in range(len(X))])))
    res["E5_abs_spearman_vs_orig"] = e5
    # E6 deletion / insertion for Grad-CAM and IG (regression: normalised output drift towards baseline)
    steps = np.linspace(0, 1, 11); b = baseline_like(x)
    with torch.no_grad():
        f_full, f_base = m0(x).numpy(), m0(b).numpy()
    denom = np.where(np.abs(f_full - f_base) > 1e-6, f_full - f_base, np.nan)
    e6 = {}
    for meth in ("gradcam", "ig"):
        for order in ("attr", "rand"):
            dl, il = [], []
            for s in steps:
                k = int(round(s * S.T_IN * 13))
                xd, xi = x.clone(), b.clone()
                for i in range(len(X)):
                    r = np.argsort(-A[(meth, 0)][i].ravel()) if order == "attr" else RNG.permutation(S.T_IN * 13)
                    tt, jj = np.unravel_index(r[:k], (S.T_IN, 13))
                    xd[i, :, tt, jj] = b[i, :, tt, jj]; xi[i, :, tt, jj] = x[i, :, tt, jj]
                with torch.no_grad():
                    dl.append((m0(xd).numpy() - f_base) / denom); il.append((m0(xi).numpy() - f_base) / denom)
            e6[f"{meth}|{order}"] = dict(deletion_auc=float(np.nanmean(np.trapz(np.stack(dl, 1), steps, axis=1))),
                                         insertion_auc=float(np.nanmean(np.trapz(np.stack(il, 1), steps, axis=1))))
    res["E6"] = e6
    res["n"] = len(X)
    summary[set_name] = res
    np.savez_compressed(OUT / f"attr_{set_name.split('(')[0]}.npz", X=X, pads=P, tags=np.array(tags),
                        **{f"{k[0]}_f{k[1]}": v for k, v in A.items()})
    # figure: 3 examples, joint x time maps
    ex = list(range(min(3, len(X))))
    fig, axes = plt.subplots(len(ex), 4, figsize=(12, 2.4 * len(ex)), squeeze=False)
    for r_, i in enumerate(ex):
        axes[r_, 0].plot(X[i][1].T[[11, 12]].T); axes[r_, 0].set_title(f"{tags[i][:38]}\nankle y (L,R); pad={P[i]}", fontsize=6)
        for c_, meth in enumerate(("gradcam", "ig", "occlusion")):
            axes[r_, c_ + 1].imshow(A[(meth, 0)][i].T, aspect="auto", cmap="magma", interpolation="nearest")
            axes[r_, c_ + 1].set_yticks(range(13)); axes[r_, c_ + 1].set_yticklabels(S.JOINTS, fontsize=5)
            axes[r_, c_ + 1].set_title(meth, fontsize=7)
    fig.tight_layout()
    fp = OUT / "figures" / f"sabo_{set_name.split('(')[0]}_examples_n{len(X)}.png"
    try:
        fig.savefig(fp, dpi=110)
    except OSError as e:  # file may be locked by an image viewer
        log("figure save failed", fp.name, repr(e))
    plt.close(fig)
    log(set_name, "done", f"{time.time()-t0:.0f}s", json.dumps({k: res[k] for k in ("gradcam", "fold_agreement_gradcam_spearman")}))

summary["E0_carepd_external_ALL"] = {k: v for k, v in ext.items() if k.endswith("|ALL")}
summary["runtime_s"] = round(time.time() - t0, 1); summary["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(summary, open(OUT / "sabo_summary.json", "w"), indent=1)
(OUT / "sabo_log.txt").write_text("\n".join(LOG), encoding="utf-8")
MG.report(" sabo_xai")
