"""Presentation animation: 2D skeleton walking, joints coloured/sized by per-frame attribution,
3 panels (Grad-CAM | Integrated Gradients | Occlusion) for the Sabo ST-GCN (public weights).
Attributions are recomputed for the selected walks and AVERAGED OVER ALL 10 FOLDS
(each fold's map is normalised to sum 1 before averaging). Prediction = 10-fold ensemble on the same
120-frame window. Headless (Agg), CPU 4 threads, 6 GB guard.
Run: MPLBACKEND=Agg .venv/Scripts/python.exe scripts/sabo_skeleton_anim.py"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, cm, colors  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from xaipd import sabo as S  # noqa: E402
from xaipd.common import RESULTS  # noqa: E402

MG.limit_torch()
t0 = time.time()
GM = RESULTS / "gait_motion"; OUT = GM / "animations"; OUT.mkdir(parents=True, exist_ok=True)
models = [S.load(p) for p in S.checkpoints("detectron")]
BONES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
ARMS, LEGS = [3, 4, 5, 6], [9, 10, 11, 12]
HIP = (100 / S.RES[0] - 0.5, 100 / S.RES[1] - 0.5)


def baseline(x):
    b = torch.zeros_like(x); b[:, 0] = HIP[0]; b[:, 1] = HIP[1]; return b


def gradcam(m, x):
    x = x.clone().requires_grad_(True); out = m(x)
    g, = torch.autograd.grad(out.sum(), m.encoder.feat); a = m.encoder.feat.detach()
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    return torch.nn.functional.interpolate(cam, size=(S.T_IN, 13), mode="bilinear", align_corners=False)[:, 0].numpy()


def ig(m, x, steps=32):
    b = baseline(x); tot = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (b + a * (x - b)).requires_grad_(True)
        g, = torch.autograd.grad(m(xi).sum(), xi); tot += g
    return ((x - b) * tot / steps).abs().sum(1).detach().numpy()


def occlusion(m, x, win=10):
    b = baseline(x); out = np.zeros((x.shape[0], S.T_IN, 13), np.float32)
    with torch.no_grad():
        s0 = m(x)
        for j in range(13):
            for t in range(0, S.T_IN, win):
                xo = x.clone(); xo[:, :, t:t + win, j] = b[:, :, t:t + win, j]
                out[:, t:t + win, j] = (s0 - m(xo)).abs().numpy()[:, None]
    return out


def fold_avg(fn, x):
    acc = 0
    for m in models:
        a = np.maximum(fn(m, x), 0)
        acc = acc + a / np.maximum(a.sum(axis=(1, 2), keepdims=True), 1e-12)
    return acc / len(models)


# ---------------- select walks
ps = np.load(GM / "attr_public_samples.npz"); cp = np.load(GM / "attr_carepd_frontal.npz")
sel = []
for tag_part in ("ID_37", "ID_48"):
    i = [k for k, t in enumerate(ps["tags"]) if tag_part in t][0]
    sel.append(dict(src="Sabo public sample", tag=str(ps["tags"][i]).replace("sample:", ""), x=ps["X"][i], pad=int(ps["pads"][i]), gt=0))
# CARE-PD: a GT=2 walk without padding where the ensemble under-calls severity (most informative contrast)
X = torch.from_numpy(cp["X"])
with torch.no_grad():
    ens = np.mean([m(X).numpy() for m in models], 0)
gts = np.array([int(t.split(":")[-1]) for t in cp["tags"]])
cand = [k for k in range(len(gts)) if gts[k] >= 2 and cp["pads"][k] == 0]
k = cand[int(np.argmin(ens[cand]))]
sel.append(dict(src="CARE-PD (frontal 2D projection, external)", tag=str(cp["tags"][k]).rsplit(":", 1)[0], x=cp["X"][k], pad=int(cp["pads"][k]), gt=int(gts[k])))

meta = []
for s in sel:
    x = torch.from_numpy(s["x"][None])
    with torch.no_grad():
        raws = np.array([float(m(x)[0]) for m in models])
    s["raw"] = float(raws.mean()); s["pred"] = int(np.clip(round(s["raw"]), 0, 2))
    s["fold_preds"] = np.clip(np.round(raws), 0, 2).astype(int).tolist()
    s["A"] = {"Grad-CAM": fold_avg(gradcam, x)[0], "Integrated Gradients": fold_avg(ig, x)[0], "Occlusion": fold_avg(occlusion, x)[0]}
    s["shares"] = {mth: dict(arms=float(a[:, ARMS].sum() / a.sum()), legs=float(a[:, LEGS].sum() / a.sum()),
                             padding=float(a[:s["pad"]].sum() / a.sum())) for mth, a in s["A"].items()}
    meta.append({k_: v for k_, v in s.items() if k_ not in ("x", "A")})
    print(s["tag"], "gt", s["gt"], "pred", s["pred"], {m_: {k2: round(v2, 2) for k2, v2 in v.items()} for m_, v in s["shares"].items()},
          f"{time.time()-t0:.0f}s", flush=True)
json.dump(meta, open(OUT / "animation_meta.json", "w"), indent=1)

# ---------------- rendering
CMAP = matplotlib.colormaps["inferno"]


def to_px(x):  # (2,T,13) model space -> (T,13,2) pixel frame (y down)
    return np.stack([(x[0] + 0.5) * S.RES[0], (x[1] + 0.5) * S.RES[1]], -1)


def draw_frame(axes, tl_axes, s, t, cursors):
    P = to_px(s["x"]); real = slice(s["pad"], S.T_IN)
    Pr = P[real]; lo, hi = Pr.reshape(-1, 2).min(0), Pr.reshape(-1, 2).max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.15
    for ax, (mth, A) in zip(axes, s["A"].items()):
        ax.cla(); ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] + r, c[1] - r); ax.set_aspect("equal"); ax.axis("off")
        vmax = A.max()
        sh = s["shares"][mth]
        ax.set_title(f"{mth}\narms {sh['arms']*100:.0f}% | legs {sh['legs']*100:.0f}% | padding {sh['padding']*100:.0f}%", fontsize=8)
        if t < s["pad"]:
            ax.text(c[0], c[1], f"PADDING\n(zero input)\nattribution here:\n{A[t].sum()/A.sum()*100:.2f}% of total / frame",
                    ha="center", va="center", fontsize=8, color="crimson")
            ax.add_patch(plt.Rectangle((c[0] - r, c[1] - r), 2 * r, 2 * r, fill=False, ec="crimson", lw=2, ls="--"))
            continue
        p = P[t]
        for a_, b_ in BONES:
            ax.plot([p[a_, 0], p[b_, 0]], [p[a_, 1], p[b_, 1]], c="0.6", lw=2, zorder=1)
        v = A[t] / vmax if vmax > 0 else A[t]
        ax.scatter(p[:, 0], p[:, 1], s=20 + 380 * v, c=CMAP(0.15 + 0.85 * v), edgecolors="k", linewidths=0.5, zorder=2)
    for ax_t, (mth, A), cur in zip(tl_axes, s["A"].items(), cursors):
        cur.set_xdata([t, t])


def render(s, idx):
    fig = plt.figure(figsize=(12, 6.2), dpi=90)
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 1.1], hspace=0.25, wspace=0.05, top=0.86, bottom=0.07, left=0.03, right=0.97)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    tl = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cursors = []
    for ax_t, (mth, A) in zip(tl, s["A"].items()):
        ax_t.plot(A[:, ARMS].sum(1), c="tab:blue", lw=1, label="arms"); ax_t.plot(A[:, LEGS].sum(1), c="tab:green", lw=1, label="legs")
        ax_t.plot(A.sum(1), c="k", lw=0.8, label="all joints")
        if s["pad"]:
            ax_t.axvspan(0, s["pad"] - 0.5, color="crimson", alpha=0.15, label="padding")
        ax_t.set_xlim(0, S.T_IN - 1); ax_t.set_yticks([]); ax_t.tick_params(labelsize=6); ax_t.set_xlabel("frame", fontsize=7)
        cursors.append(ax_t.axvline(0, c="crimson", lw=1))
    tl[0].legend(fontsize=6, loc="upper right", ncol=2)
    head = fig.text(0.5, 0.955, "", ha="center", fontsize=10)
    fig.text(0.5, 0.915, f"{s['src']}  |  Sabo ST-GCN public weights, 10-fold mean  |  joint colour+size = |attribution| (per-sample normalised)",
             ha="center", fontsize=7, color="0.3")

    def update(t):
        draw_frame(axes, tl, s, t, cursors)
        head.set_text(f"{s['tag']}   GT UPDRS-gait = {s['gt']}   predicted = {s['pred']} (raw {s['raw']:.2f}; folds {''.join(map(str, s['fold_preds']))})"
                      f"   frame {t+1}/{S.T_IN}" + ("   [PADDING]" if t < s["pad"] else ""))
        return []

    name = f"skeleton_{idx}_{s['tag'].replace(':', '_')[:40]}"
    ani = animation.FuncAnimation(fig, update, frames=S.T_IN, blit=False)
    gif = OUT / f"{name}.gif"
    ani.save(gif, writer=animation.PillowWriter(fps=15))
    mp4 = None
    ff = shutil.which("ffmpeg")
    if ff:
        mp4 = OUT / f"{name}.mp4"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "faststart", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=False)
    plt.close(fig)
    return gif, mp4


outs = []
for i, s in enumerate(sel):
    outs.append(render(s, i + 1)); print("rendered", outs[-1][0].name, f"{time.time()-t0:.0f}s", flush=True)

# ---------------- montage: rows = walks, cols = 3 methods x 2 representative frames
fig, axs = plt.subplots(len(sel), 6, figsize=(15, 3.2 * len(sel)), dpi=100)
for r_, s in enumerate(sel):
    real = np.arange(s["pad"], S.T_IN)
    frames = [int(real[len(real) // 3]), int(real[2 * len(real) // 3])]
    P = to_px(s["x"]); Pr = P[s["pad"]:]; lo, hi = Pr.reshape(-1, 2).min(0), Pr.reshape(-1, 2).max(0); c = (lo + hi) / 2; rad = (hi - lo).max() / 2 * 1.15
    for m_i, (mth, A) in enumerate(s["A"].items()):
        for f_i, t in enumerate(frames):
            ax = axs[r_, m_i * 2 + f_i]; p = P[t]; v = A[t] / A.max()
            for a_, b_ in BONES:
                ax.plot([p[a_, 0], p[b_, 0]], [p[a_, 1], p[b_, 1]], c="0.6", lw=1.5)
            ax.scatter(p[:, 0], p[:, 1], s=15 + 250 * v, c=CMAP(0.15 + 0.85 * v), edgecolors="k", linewidths=0.4, zorder=2)
            ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] + rad, c[1] - rad); ax.set_aspect("equal"); ax.axis("off")
            sh = s["shares"][mth]
            ax.set_title(f"{mth} f{t+1}\narms {sh['arms']*100:.0f}% legs {sh['legs']*100:.0f}% pad {sh['padding']*100:.0f}%", fontsize=7)
    axs[r_, 0].text(-0.15, 0.5, f"{s['tag'][:30]}\nGT {s['gt']} / pred {s['pred']}\npad {s['pad']}/120 fr", transform=axs[r_, 0].transAxes,
                    ha="right", va="center", fontsize=7)
fig.suptitle("Sabo ST-GCN (public weights, 10-fold mean): Grad-CAM paints the arms; occlusion points at legs, IG at head/legs; padding frames also receive attribution", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "skeleton_montage.png", dpi=110); plt.close(fig)
print("montage done", f"{time.time()-t0:.0f}s")
MG.report(" skeleton_anim")
