"""Gait XAI: G2 (reproduced El Maachi 1D-CNN, 3 seeds) + G1 (Naimi 2023 public pretrained ConvTransformer
weights, actually transformer-only severity model). CPU only.
Metrics mirror the drawings pipeline: ROI(gait-event) enrichment vs random, method agreement,
seed/model agreement, E5 cascading randomisation, E6 deletion/insertion.
Run: .venv/Scripts/python.exe scripts/gait_xai.py"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xaipd import memguard as MG  # noqa: E402  (before numpy/torch)
import numpy as np
import torch
from safetensors.torch import load_file

from xaipd import gait as G
from xaipd.common import RESULTS, ROOT, spearman, write_csv

MG.limit_torch()
OUT = RESULTS / "gait"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
STEPS = np.linspace(0, 1, 11)
NXAI = 300

t0 = time.time()
d = G.build_dataset(ROOT / "data" / "processed" / "gaitpdb_seg.npz")
split = json.load(open(OUT / "subject_split.json"))
te_mask = np.isin(d["subj"], split["test"])
TE = np.where(te_mask)[0]
XI = np.sort(RNG.choice(TE, NXAI, replace=False))
X = torch.from_numpy(d["X"][XI]); RAW = d["R"][XI]; Y = d["y"][XI]
ROI = np.stack([G.gait_event_mask(r) for r in RAW])


def auc(y):
    return float(np.trapz(y, STEPS))


def del_ins(score_fn, x, attr):
    """x (B,100,18) torch; attr (B,100,18). zero = 'no force' baseline."""
    B = x.shape[0]; n = 100 * 18
    out = {}
    for tag, ranks in (("cam", np.argsort(-attr.reshape(B, -1), 1)),
                       ("rand", np.stack([RNG.permutation(n) for _ in range(B)]))):
        dc, ic = [], []
        for s in STEPS:
            k = int(round(s * n))
            m = np.zeros((B, n), bool)
            np.put_along_axis(m, ranks[:, :k], True, 1)
            m = torch.from_numpy(m.reshape(B, 100, 18))
            dc.append(score_fn(x.masked_fill(m, 0.0)))
            ic.append(score_fn(torch.where(m, x, torch.zeros_like(x))))
        out[f"del_{tag}"] = float(np.mean([auc(c) for c in np.stack(dc, 1)]))
        out[f"ins_{tag}"] = float(np.mean([auc(c) for c in np.stack(ic, 1)]))
    return out


def roi_summary(attr):
    rows = [G.roi_metrics(attr[i], ROI[i])[0] for i in range(len(attr))]
    s = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    per_sensor = np.nanmean([G.roi_metrics(attr[i], ROI[i])[1] for i in range(len(attr))], 0)
    return s, per_sensor


def pair_rho(a, b):
    return float(np.nanmean([spearman(a[i], b[i]) for i in range(len(a))]))


results, attrs, sensor_share = {}, {}, {}

# ---------------- G2: reproduced 1D-CNN (3 seeds) ----------------
models = {}
for seed in (0, 1, 2):
    m = G.ParallelCNN(); m.load_state_dict(load_file(str(ROOT / "models" / "safe" / "g2_elmaachi_repro" / f"seed{seed}.safetensors")))
    models[seed] = m.eval()
for seed, m in models.items():
    with torch.no_grad():
        pred = (torch.sigmoid(m(X)) >= 0.5).float()
    sign = pred * 2 - 1
    score = lambda xx, m=m, sign=sign: (m(xx) * sign)  # logit of predicted class
    prob = lambda xx, m=m, pred=pred: (lambda p: (p * pred + (1 - p) * (1 - pred)).numpy())(torch.sigmoid(m(xx).detach()))
    a = {"gradcam": G.gradcam_1d(m, X), "ig": G.integrated_gradients(score, X), "occlusion": G.occlusion(lambda xx: score(xx).detach(), X)}
    for k, v in a.items():
        attrs[(f"G2_seed{seed}", k)] = v
        s, ps = roi_summary(v)
        results[f"G2_seed{seed}|{k}"] = s
        sensor_share[f"G2_seed{seed}|{k}"] = ps
    results[f"G2_seed{seed}|E6_gradcam"] = del_ins(prob, X, a["gradcam"])
    results[f"G2_seed{seed}|E6_ig"] = del_ins(prob, X, a["ig"])
    results[f"G2_seed{seed}|method_agreement"] = {
        "gradcam-vs-ig": pair_rho(a["gradcam"], a["ig"]), "gradcam-vs-occlusion": pair_rho(a["gradcam"], a["occlusion"]),
        "ig-vs-occlusion": pair_rho(a["ig"], a["occlusion"])}
    print("G2 seed", seed, f"{time.time()-t0:.0f}s", flush=True)

for k in ("gradcam", "ig", "occlusion"):
    results[f"G2|seed_agreement|{k}"] = {f"{i}-vs-{j}": pair_rho(attrs[(f'G2_seed{i}', k)], attrs[(f'G2_seed{j}', k)])
                                         for i, j in ((0, 1), (0, 2), (1, 2))}

# E5 cascading randomisation on seed0 (head -> c4 -> c3 -> all)
m0 = models[0]; sd = {k: v.clone() for k, v in m0.state_dict().items()}
with torch.no_grad():
    sign0 = ((torch.sigmoid(m0(X)) >= 0.5).float() * 2 - 1)
g = torch.Generator().manual_seed(0)
stages = {"head": lambda m: [m.head] + [b.fc for b in m.branches],
          "head+conv4": lambda m: [m.head] + [b.fc for b in m.branches] + [b.c4 for b in m.branches],
          "head+conv3-4": lambda m: [m.head] + [b.fc for b in m.branches] + [b.c4 for b in m.branches] + [b.c3 for b in m.branches],
          "all": lambda m: [m]}
e5 = {}
for name, sel in stages.items():
    with torch.no_grad():
        for mod in sel(m0):
            for p in mod.parameters():
                p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g)].view_as(p))
    # Grad-CAM must explain the ORIGINAL predicted class -> use sign0 via IG; for Grad-CAM use helper on score
    gc = G.gradcam_1d(m0, X)
    ig = G.integrated_gradients(lambda xx: m0(xx) * sign0, X)
    e5[name] = {"gradcam": float(np.nanmean([abs(spearman(attrs[('G2_seed0', 'gradcam')][i], gc[i])) for i in range(NXAI)])),
                "ig": float(np.nanmean([abs(spearman(attrs[('G2_seed0', 'ig')][i], ig[i])) for i in range(NXAI)]))}
    m0.load_state_dict(sd)
results["G2_seed0|E5_abs_spearman_vs_orig"] = e5
print("E5 done", f"{time.time()-t0:.0f}s", flush=True)

# ---------------- G1: Naimi 2023 public weights (Keras, severity 4-class, no conv layers) ----------------
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU"); MG.limit_tf()
    g1 = tf.keras.models.load_model(str(ROOT / "models" / "raw" / "g1_naimi_convtransformer" / "main.hdf5"), compile=False)

    def g1_prob_pd(xb):  # xb np (B,100,18)
        out = []
        for i in range(0, len(xb), 512):
            b = xb[i:i + 512]
            out.append(1 - g1([b[:, :, c:c + 1] for c in range(18)], training=False).numpy()[:, 0])
        return np.concatenate(out)

    # E0 on all test-subject segments (CAVEAT: training subjects of this checkpoint unknown -> leakage)
    pte = g1_prob_pd(d["X"][TE]); yte = d["y"][TE]; ste = d["subj"][TE]
    us = np.unique(ste)
    sp = np.array([pte[ste == s].mean() for s in us]); sy = np.array([yte[ste == s][0] for s in us])
    from sklearn.metrics import roc_auc_score
    results["G1_naimi|E0_on_our_test_subjects"] = dict(seg_acc=float(((pte >= 0.5) == yte).mean()),
                                                     subj_acc=float(((sp >= 0.5) == sy).mean()), subj_auc=float(roc_auc_score(sy, sp)),
                                                     note="checkpoint trained on unknown fold of the same 166 subjects: likely leakage")
    Xn = X.numpy(); p1 = g1_prob_pd(Xn); pred1 = (p1 >= 0.5).astype(np.float32)

    def g1_score_tf(xt):  # prob of predicted class, differentiable
        p = 1 - g1([xt[:, :, c:c + 1] for c in range(18)], training=False)[:, 0]
        return p * pred1 + (1 - p) * (1 - pred1)

    def g1_ig(steps=32):
        xt = tf.constant(Xn); tot = tf.zeros_like(xt)
        for a in np.linspace(1 / steps, 1, steps):
            xi = tf.Variable(a * xt)
            with tf.GradientTape() as tape:
                s = tf.reduce_sum(g1_score_tf(xi))
            tot += tape.gradient(s, xi)
        return np.abs((xt * tot / steps).numpy())

    def g1_occ(win=10):
        s0 = g1_score_tf(tf.constant(Xn)).numpy(); out = np.zeros_like(Xn)
        for c in range(18):
            for t in range(0, 100, win):
                xo = Xn.copy(); xo[:, t:t + win, c] = 0
                out[:, t:t + win, c] = np.clip(s0 - g1_score_tf(tf.constant(xo)).numpy(), 0, None)[:, None]
        return out

    a1 = {"ig": g1_ig(), "occlusion": g1_occ()}
    for k, v in a1.items():
        attrs[("G1_naimi", k)] = v
        s, ps = roi_summary(v); results[f"G1_naimi|{k}"] = s; sensor_share[f"G1_naimi|{k}"] = ps
    results["G1_naimi|method_agreement"] = {"ig-vs-occlusion": pair_rho(a1["ig"], a1["occlusion"])}
    results["G1_naimi|E6_ig"] = del_ins(lambda xx: g1_score_tf(tf.constant(xx.numpy())).numpy(), X, a1["ig"])
    for k in ("ig", "occlusion"):
        results[f"G1-vs-G2seed0|{k}"] = pair_rho(attrs[("G1_naimi", k)], attrs[("G2_seed0", k)])
    print("G1 done", f"{time.time()-t0:.0f}s", flush=True)
except Exception as e:  # report, do not hide
    results["G1_naimi|error"] = repr(e)
    print("G1 failed:", repr(e), flush=True)

# random / oracle baselines on the same segments
rnd = RNG.random((NXAI, 100, 18)).astype(np.float32)
results["baseline|random"] = roi_summary(rnd)[0]
results["baseline|oracle_roi"] = roi_summary(ROI.astype(np.float32))[0]
results["meta"] = dict(n_xai_segments=NXAI, n_test_subjects=len(split["test"]), runtime_s=round(time.time() - t0, 1))
json.dump(results, open(OUT / "gait_xai.json", "w"), indent=1)
write_csv(OUT / "per_sensor_share.csv", [dict(key=k, **{G.CH_NAMES[i]: float(v[i]) for i in range(18)}) for k, v in sensor_share.items()])
np.savez_compressed(OUT / "attributions.npz", idx=XI, **{f"{a}__{b}": v for (a, b), v in attrs.items()})

# ---------------- figures ----------------
ex = [int(np.where(Y == 1)[0][0]), int(np.where(Y == 0)[0][0])]
for j, i in enumerate(ex):
    keys = [k for k in attrs if k[0] in ("G2_seed0", "G2_seed1", "G1_naimi")]
    fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(9, 1.6 * (len(keys) + 1)), sharex=True)
    axes[0].plot(RAW[i][:, 16], label="L total"); axes[0].plot(RAW[i][:, 17], label="R total")
    for t in np.where(ROI[i][:, 16] | ROI[i][:, 17])[0]:
        axes[0].axvspan(t - 0.5, t + 0.5, color="grey", alpha=0.15, lw=0)
    axes[0].legend(fontsize=6); axes[0].set_title(f"segment y={Y[i]} (grey = gait-event ROI)", fontsize=8)
    for ax, k in zip(axes[1:], keys):
        ax.imshow(attrs[k][i].T, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_yticks([0, 8, 16]); ax.set_yticklabels(["L1", "R1", "Ltot"], fontsize=6)
        ax.set_title(f"{k[0]} {k[1]}", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"segment_example_{j}.png", dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 3))
for k in [k for k in sensor_share if k.endswith("|ig") or k.endswith("|gradcam")]:
    ax.plot(sensor_share[k], marker="o", ms=3, label=k)
ax.axhline(1 / 18, ls="--", c="k", lw=0.8, label="uniform")
ax.set_xticks(range(18)); ax.set_xticklabels(G.CH_NAMES, fontsize=7); ax.legend(fontsize=6); ax.set_ylabel("attribution share")
fig.tight_layout(); fig.savefig(OUT / "figures" / "per_sensor_share.png", dpi=110); plt.close(fig)
print(json.dumps(results, indent=1))
print("total", f"{time.time()-t0:.0f}s")

MG.report(" gait_xai")
