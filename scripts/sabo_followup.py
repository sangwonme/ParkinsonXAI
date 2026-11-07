"""Follow-ups for the Sabo ST-GCN (CPU, 4 threads, 6 GB guard):
(a) group occlusion: does the prediction actually depend on arms vs legs?
(b) padding shortcut: same walk, different padding (zero-begin [original], zero-end, edge-replicate,
    tile/repeat, no padding) and different pad lengths -> prediction shift (10 folds)
(c) input-scale sensitivity curve on CARE-PD (frontal view, torso length 100..260), stratified subset.
Run: .venv/Scripts/python.exe scripts/sabo_followup.py"""
from __future__ import annotations

import json
import os
import sys
import time
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
from xaipd.common import RESULTS, spearman  # noqa: E402

MG.limit_torch()
OUT = RESULTS / "gait_motion"; RNG = np.random.default_rng(0); t0 = time.time()
models = [S.load(p) for p in S.checkpoints("detectron")]
HIP = (100 / S.RES[0] - 0.5, 100 / S.RES[1] - 0.5)
ARMS, LEGS = [3, 4, 5, 6], [9, 10, 11, 12]


def ens(x):  # x (N,2,T,13) -> (folds, N) raw outputs
    with torch.no_grad():
        return np.stack([np.concatenate([m(torch.from_numpy(x[i:i + 512])).numpy() for i in range(0, len(x), 512)]) for m in models])


def pad(x, mode, L=S.T_IN):
    """x (2,T,13) real frames."""
    C, T, V = x.shape
    if mode == "none" or T >= L:
        return x
    if mode == "zero_begin":
        o = np.zeros((C, L, V), np.float32); o[:, L - T:] = x; return o
    if mode == "zero_end":
        o = np.zeros((C, L, V), np.float32); o[:, :T] = x; return o
    if mode == "edge":
        return np.concatenate([np.repeat(x[:, :1], L - T, 1), x], 1)
    if mode == "tile":
        return np.tile(x, (1, int(np.ceil(L / T)), 1))[:, :L]
    raise ValueError(mode)


res = {}
# ------------------------------------------------ (b) padding on public samples (all short walks)
samples = [S.load_csv(f) for f in sorted((S.DATA / "tri_detectron").rglob("*.csv"))]
pb = {}
for mode in ("zero_begin", "zero_end", "edge", "tile", "none"):
    outs = [ens(pad(d["x"], mode)[None])[:, 0] for d in samples]  # each (10,)
    pb[mode] = dict(mean_raw=[round(float(o.mean()), 3) for o in outs], per_fold_pred_class=[np.clip(np.round(o), 0, 2).astype(int).tolist() for o in outs])
pb["walk_T"] = [d["T"] for d in samples]; pb["true_updrs"] = [d["updrs"] for d in samples]
res["padding_public_samples"] = pb

# ------------------------------------------------ CARE-PD walks (frontal, torso 190)
J, nose = CP.smpl_rest()
walks = []
for ds in CP.UPDRS_SETS:
    for w in CP.iter_walks(ds):
        if w["updrs"] is not None:
            walks.append(dict(ds=ds, y=min(int(w["updrs"]), 2), X3=CP.fk_coco13(w["pose"], w["trans"], J, nose)))
# stratified subset: up to 100 walks per (dataset, class)
idx = []
for ds in CP.UPDRS_SETS:
    for c in (0, 1, 2):
        ii = [i for i, w in enumerate(walks) if w["ds"] == ds and w["y"] == c]
        idx += list(RNG.choice(ii, min(100, len(ii)), replace=False)) if ii else []
sub = [walks[i] for i in idx]
y = np.array([w["y"] for w in sub])
print("subset", len(sub), {c: int((y == c).sum()) for c in (0, 1, 2)}, f"{time.time()-t0:.0f}s", flush=True)


def mid_window(x, L=S.T_IN):
    T = x.shape[1]
    if T <= L:
        return pad(x, "zero_begin")
    s = (T - L) // 2
    return x[:, s:s + L]


Xf = np.stack([mid_window(S.preprocess_xyconf(CP.to_sabo_frame(CP.VIEWS["frontal"](w["X3"]), 190.0))) for w in sub])

# (b2) padding on CARE-PD: keep only the last k real frames, pad the rest in different ways
pc = {}
for k in (40, 60, 90):
    base = None
    for mode in ("zero_begin", "edge", "tile", "none"):
        Xk = np.stack([pad(x[:, -k:], mode) for x in Xf]) if mode != "none" else None
        o = ens(Xk) if mode != "none" else np.stack([ens(x[None, :, -k:])[:, 0] for x in Xf], 1)
        mr = o.mean(0)
        if mode == "zero_begin":
            base = mr
        pc[f"k{k}|{mode}"] = dict(mean_raw=float(mr.mean()), spearman_vs_updrs=float(spearman(mr, y)),
                                  mean_abs_shift_vs_zero_begin=float(np.abs(mr - base).mean()),
                                  class_change_rate_vs_zero_begin=float((np.clip(np.round(mr), 0, 2) != np.clip(np.round(base), 0, 2)).mean()))
    print("pad k", k, f"{time.time()-t0:.0f}s", flush=True)
full = ens(Xf).mean(0)
pc["full120_no_pad"] = dict(mean_raw=float(full.mean()), spearman_vs_updrs=float(spearman(full, y)))
res["padding_carepd"] = pc

# ------------------------------------------------ (a) group occlusion (arms vs legs vs others) on CARE-PD + samples
def occlude(X, joints):
    Xo = X.copy(); Xo[:, 0, :, joints] = HIP[0]; Xo[:, 1, :, joints] = HIP[1]; return Xo


go = {}
for name, X in (("carepd_frontal", Xf), ("public_samples", np.stack([pad(d["x"], "zero_begin") for d in samples]))):
    o0 = ens(X).mean(0)
    go[name] = {g: dict(mean_abs_change=float(np.abs(ens(occlude(X, jj)).mean(0) - o0).mean()),
                        class_change_rate=float((np.clip(np.round(ens(occlude(X, jj)).mean(0)), 0, 2) != np.clip(np.round(o0), 0, 2)).mean()))
                for g, jj in (("arms", ARMS), ("legs", LEGS), ("head+shoulders", [0, 1, 2]), ("hips", [7, 8]))}
    go[name]["mean_abs_output"] = float(np.abs(o0).mean())
res["group_occlusion"] = go
print("group occlusion", json.dumps(go), flush=True)

# ------------------------------------------------ (c) scale sensitivity curve
torsos = [100, 130, 150, 170, 190, 210, 230, 260]
sc = {}
for tl in torsos:
    X = np.stack([mid_window(S.preprocess_xyconf(CP.to_sabo_frame(CP.VIEWS["frontal"](w["X3"]), float(tl)))) for w in sub])
    mr = ens(X).mean(0); p = np.clip(np.round(mr), 0, 2)
    sc[tl] = dict(mean_raw=float(mr.mean()), spearman=float(spearman(mr, y)), acc=float((p == y).mean()),
                  pred_dist={str(c): int((p == c).sum()) for c in (0, 1, 2)})
    print("torso", tl, sc[tl], flush=True)
res["scale_sensitivity_frontal"] = sc
res["subset_n"] = len(sub); res["runtime_s"] = round(time.time() - t0, 1); res["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(res, open(OUT / "sabo_followup.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(torsos, [sc[t]["spearman"] for t in torsos], "o-"); ax[0].axhline(0, c="k", lw=0.7)
ax[0].axvspan(153, 217, color="grey", alpha=0.15, label="Sabo sample torso range")
ax[0].set_xlabel("torso length used for normalisation (units)"); ax[0].set_ylabel("Spearman(pred, UPDRS-gait)"); ax[0].legend(fontsize=7)
ax[1].plot(torsos, [sc[t]["mean_raw"] for t in torsos], "o-", c="C1"); ax[1].set_xlabel("torso length"); ax[1].set_ylabel("mean predicted score")
fig.suptitle("Sabo ST-GCN on CARE-PD (frontal 2D projection): input-scale sensitivity", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "figures" / "sabo_scale_sensitivity.png", dpi=120); plt.close(fig)
MG.report(" sabo_followup")
