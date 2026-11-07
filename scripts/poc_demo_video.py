"""Demo video (headless): the same walk shown side by side.
Left  = black-box joint x time CNN (reproduced training) + Grad-CAM joint colouring (baseline).
Right = proposed symptom model: same skeleton + per-symptom contribution bars + a plain-English sentence.
Segments: (1) original healthy walk, (2) same walk with reduced arm swing injected, (3) with shorter stride.
Fold-0 models; the walk is from the fold-0 TEST set and is chosen for visibility (disclosed in the video).
Output: results/poc/demo/poc_demo.mp4 (<10 MB) + poc_demo_preview.gif
Run: MPLBACKEND=Agg .venv/Scripts/python.exe scripts/poc_demo_video.py"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["MPLBACKEND"] = "Agg"; os.environ.setdefault("XAIPD_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from xaipd import symptoms as SY  # noqa: E402
from xaipd.common import RESULTS, ROOT  # noqa: E402

MG.limit_torch()
OUT = RESULTS / "poc" / "demo"; OUT.mkdir(parents=True, exist_ok=True)
BONES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
FPS_OUT, NFR, CLIP, STRIDE = 15, 180, 60, 30
CMAP = matplotlib.colormaps["inferno"]


class JTCNN(nn.Module):  # identical to scripts/poc_blackbox.py
    def __init__(self):
        super().__init__()
        c = lambda i, o, k, s=(1, 1): nn.Sequential(nn.Conv2d(i, o, k, s, ((k[0] - 1) // 2, 0)), nn.BatchNorm2d(o), nn.ReLU())
        self.t1, self.t2, self.t3 = c(3, 32, (9, 1)), c(32, 64, (9, 1), (2, 1)), c(64, 64, (9, 1))
        self.mix = c(64, 128, (1, 13)); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3)); self.feat = None

    def forward(self, x):
        x = self.t3(self.t2(self.t1(x))); self.feat = x
        return self.fc(self.mix(x).mean(dim=(2, 3)))


def prep(w):
    hip = (w[:, SY.J_LHIP] + w[:, SY.J_RHIP]) / 2
    return (w - hip[:, None]).astype(np.float32)


def bb_frame_cam(m, w):
    """Per-frame joint Grad-CAM (expected-severity target), averaged over overlapping 60-frame clips."""
    x = prep(w); T = len(x); acc = np.zeros((T, 13)); cnt = np.zeros((T, 1)); probs = []
    sev = torch.arange(3, dtype=torch.float32)
    for s in range(0, max(T - CLIP, 0) + 1, STRIDE):
        c = torch.from_numpy(x[s:s + CLIP].transpose(2, 0, 1)[None]).requires_grad_(True)
        p = torch.softmax(m(c), -1); probs.append(p.detach().numpy()[0])
        g, = torch.autograd.grad((p * sev).sum(), m.feat); a = m.feat.detach()
        cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))[0]  # (30,13)
        cam = torch.nn.functional.interpolate(cam[None, None], size=(CLIP, 13), mode="bilinear", align_corners=False)[0, 0].numpy()
        acc[s:s + CLIP] += cam; cnt[s:s + CLIP] += 1
    return acc / np.maximum(cnt, 1), np.mean(probs, 0)


def project(X):  # oblique view (45 deg) so both arm swing and stride are visible; y up -> image down
    f = SY.forward_axis(X); up = np.array([0, 1.0, 0]); lat = np.cross(up, f)
    u = (X @ (0.7071 * f + 0.7071 * lat)); v = -(X @ up)
    u = u - ((u[:, 7] + u[:, 8]) / 2)[:, None]  # follow the walker (treadmill view) so the skeleton fills the panel
    return np.stack([u, v], -1)


walks, meta = SY.load_cache(); y = meta["y"]; groups = meta["subj"]
split = json.load(open(RESULTS / "poc" / "splits.json"))
te0 = [i for i in range(len(walks)) if groups[i] in set(split["folds"][0]["test_subjects"])]
sym = SY.SymptomModel(); sym.load_state_dict(load_file(str(ROOT / "models" / "safe" / "poc_symptom" / "fold0.safetensors"))); sym.eval()
bb = JTCNN(); bb.load_state_dict(load_file(str(ROOT / "models" / "safe" / "poc_blackbox" / "fold0.safetensors"))); bb.eval()


def sym_out(w):
    with torch.no_grad():
        o = sym(SY.batchify([w], [0])); p = sym.class_probs(o["yhat"])[0].numpy()
    return o["e"][0].numpy(), float(o["r"][0]), p


# choose a long healthy test walk that both models call 0 and whose arm-swing injection is visible
cands = [i for i in te0 if y[i] == 0 and len(walks[i]) >= NFR + 10]
best, best_i = -1e9, None
for i in cands[:80]:
    e0, _, p0 = sym_out(walks[i]); _, pb0 = bb_frame_cam(bb, walks[i])
    if p0.argmax() != 0 or pb0.argmax() != 0:
        continue
    e1, _, _ = sym_out(SY.inject(walks[i], "reduced_arm_swing"))
    d = e1[SY.SYMPTOMS.index("arm_swing")] - e0[SY.SYMPTOMS.index("arm_swing")]
    if d > best:
        best, best_i = d, i
w0 = walks[best_i]
segments = [("Original healthy walk (UPDRS-gait 0)", w0),
            ("Same walk, arm swing reduced to 30% (synthetic)", SY.inject(w0, "reduced_arm_swing")),
            ("Same walk, stride shortened to 55% (synthetic)", SY.inject(w0, "short_stride"))]
info = []
for title, w in segments:
    cam, pb = bb_frame_cam(bb, w); e, r, p = sym_out(w)
    info.append(dict(title=title, w=w, cam=cam, pb=pb, e=e, r=r, p=p, P2=project(w)))
cam_max = max(s["cam"].max() for s in info)
emax = max(np.abs(s["e"]).max() for s in info) * 1.15 + 1e-6


def sentence(e):
    order = np.argsort(-np.abs(e))[:2]
    parts = []
    for k in order:
        if abs(e[k]) < 0.05:
            continue
        name = SY.NICE[SY.SYMPTOMS[k]]
        parts.append(f"{name} {'raises' if e[k] > 0 else 'lowers'} the score by {abs(e[k]):.2f}")
    return "; ".join(parts) if parts else "all symptoms close to the healthy reference"


fig = plt.figure(figsize=(10.24, 5.76), dpi=100)
axL = fig.add_axes([0.02, 0.17, 0.30, 0.66]); axR = fig.add_axes([0.35, 0.17, 0.25, 0.66]); axB = fig.add_axes([0.73, 0.2, 0.25, 0.6])
top = fig.text(0.5, 0.95, "", ha="center", fontsize=12, weight="bold")
sub = fig.text(0.5, 0.06, "", ha="center", fontsize=10)
note = fig.text(0.5, 0.015, "CARE-PD walk from a held-out fold (example chosen for visibility). Both models trained on the same subjects.",
                ha="center", fontsize=7, color="0.4")
frames = [(si, t) for si in range(len(info)) for t in range(NFR)]


def draw(k):
    si, t = frames[k]; s = info[si]; P = s["P2"]; T = len(P); tt = t % T
    lo, hi = P.reshape(-1, 2).min(0), P.reshape(-1, 2).max(0); c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.08
    for ax in (axL, axR):
        ax.cla(); ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] + r, c[1] - r); ax.set_aspect("equal"); ax.axis("off")
        for a_, b_ in BONES:
            ax.plot([P[tt, a_, 0], P[tt, b_, 0]], [P[tt, a_, 1], P[tt, b_, 1]], c="0.6", lw=2)
    v = s["cam"][tt] / cam_max
    axL.scatter(P[tt, :, 0], P[tt, :, 1], s=20 + 300 * v, c=CMAP(0.15 + 0.85 * v), edgecolors="k", linewidths=0.4, zorder=3)
    axL.set_title(f"Baseline: black-box CNN + Grad-CAM\npredicted class {s['pb'].argmax()} (expected {s['pb'] @ np.arange(3):.2f})", fontsize=9)
    axR.scatter(P[tt, :, 0], P[tt, :, 1], s=25, c="tab:blue", zorder=3)
    axR.set_title(f"Ours: symptom model\npredicted class {s['p'].argmax()} (expected {s['p'] @ np.arange(3):.2f})", fontsize=9)
    axB.cla()
    names = [SY.NICE[k] for k in SY.SYMPTOMS] + ["unexplained part"]
    vals = list(s["e"]) + [s["r"]]
    axB.barh(range(len(vals)), vals, color=["tab:red" if x > 0 else "tab:green" for x in s["e"]] + ["0.6"])
    axB.set_yticks(range(len(vals))); axB.set_yticklabels(names, fontsize=7); axB.invert_yaxis()
    axB.set_xlim(-emax, emax); axB.axvline(0, c="k", lw=0.8); axB.tick_params(axis="x", labelsize=7)
    axB.set_title("contribution to severity score\n(red raises, green lowers)", fontsize=8)
    top.set_text(f"{si+1}/3  {s['title']}")
    if si == 0:
        sub.set_text("Ours: " + sentence(s["e"]) + ".    Grad-CAM: bright joints = where the CNN 'looks'.")
    else:
        d = s["e"] - info[0]["e"]; k = int(np.argmax(np.abs(d)))
        sub.set_text(f"Ours, change vs the original walk: {SY.NICE[SY.SYMPTOMS[k]]} {d[k]:+.2f} (largest change).   "
                     f"Grad-CAM: bright joints = where the CNN 'looks'.")
    return []


ani = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
mp4 = OUT / "poc_demo.mp4"
ff = shutil.which("ffmpeg")
if ff:
    matplotlib.rcParams["animation.ffmpeg_path"] = ff
    ani.save(mp4, writer=animation.FFMpegWriter(fps=FPS_OUT, codec="libx264", bitrate=900, extra_args=["-pix_fmt", "yuv420p", "-crf", "28"]))
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "fps=6,scale=480:-1:flags=lanczos", str(OUT / "poc_demo_preview.gif")], check=False)
else:
    ani.save(OUT / "poc_demo_preview.gif", writer=animation.PillowWriter(fps=6))
plt.close(fig)
json.dump(dict(walk=str(meta["walk"][best_i]), dataset=str(meta["ds"][best_i]), segments=[
    dict(title=s["title"], blackbox_probs=s["pb"].round(3).tolist(), symptom_probs=s["p"].round(3).tolist(),
         contributions=dict(zip(SY.SYMPTOMS, s["e"].round(3).tolist())), residual=round(s["r"], 3)) for s in info]),
    open(OUT / "demo_meta.json", "w"), indent=1)
print("saved", mp4, mp4.stat().st_size / 1e6 if mp4.exists() else None, "MB")
MG.report(" demo")
