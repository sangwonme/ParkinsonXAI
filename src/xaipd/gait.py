"""Gait (PhysioNet gaitpdb VGRF) data pipeline + reproduced El Maachi et al. 2020 parallel 1D-CNN
(PyTorch port of imanneelmaachi/.../src/algo.py) + time-series XAI and metrics.
Preprocessing follows both reference repos: 18 channels (16 sensors + 2 foot totals), 100-sample
segments (1 s @100 Hz), step 50, per-segment L2 normalisation over (time, channel)
(the repos' norm == 'std ' typo means L2 is what actually runs)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .common import ROOT

GDIR = ROOT / "data" / "raw" / "gaitpdb"
SEG, STEP, NCH = 100, 50, 18
CH_NAMES = [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)] + ["Ltot", "Rtot"]


def find_files():
    files = sorted(p for p in GDIR.rglob("*.txt") if re.match(r"(Ga|Ju|Si)(Co|Pt)\d+_\d+\.txt$", p.name))
    return files


def subject_of(p: Path) -> str:
    return p.name.split("_")[0]  # e.g. GaPt03


def label_of(p: Path) -> int:
    return int("Pt" in p.name[:4])


def load_file(p: Path) -> np.ndarray:
    a = np.loadtxt(p)
    return a[:, 1:19].astype(np.float32)  # drop time column -> (T, 18)


def segment(x: np.ndarray):
    T = x.shape[0]
    starts = range(0, T - SEG, STEP)  # same as reference: range(0, T - SEG, STEP)
    segs = np.stack([x[s:s + SEG] for s in starts])  # (n, 100, 18)
    raw = segs.copy()
    n = np.sqrt((segs ** 2).sum(axis=(1, 2), keepdims=True))
    segs = segs / np.maximum(n, 1e-12)
    return segs, raw


def build_dataset(cache: Path | None = None):
    if cache and cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    X, R, y, subj, rec = [], [], [], [], []
    for p in find_files():
        s, r = segment(load_file(p))
        X.append(s); R.append(r)
        y += [label_of(p)] * len(s)
        subj += [subject_of(p)] * len(s)
        rec += [p.stem] * len(s)
    d = dict(X=np.concatenate(X), R=np.concatenate(R), y=np.array(y, np.int64),
             subj=np.array(subj), rec=np.array(rec))
    if cache:
        np.savez(cache, **d)
    return d


def subject_split(subj, y, test_frac=0.25, seed=0):
    """Stratified hold-out at the subject level (no subject in both sets)."""
    rng = np.random.default_rng(seed)
    us = np.unique(subj)
    lab = {s: int(y[subj == s][0]) for s in us}
    test = []
    for c in (0, 1):
        ss = np.array([s for s in us if lab[s] == c])
        rng.shuffle(ss)
        test += list(ss[: int(round(len(ss) * test_frac))])
    test = set(test)
    te = np.array([s in test for s in subj])
    return ~te, te


# ---------------- model (El Maachi 2020 parallel 1D-ConvNet) ----------------
class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, 8, 3); self.c2 = nn.Conv1d(8, 16, 3)
        self.c3 = nn.Conv1d(16, 16, 3); self.c4 = nn.Conv1d(16, 16, 3)
        self.pool = nn.MaxPool1d(2); self.act = nn.SELU()
        self.fc = nn.Linear(16 * 22, 50); self.elu = nn.ELU()

    def forward(self, x):  # x (B,1,100)
        x = self.act(self.c2(self.act(self.c1(x))))
        x = self.pool(x)
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))  # (B,16,44) <- Grad-CAM target (last conv)
        x = self.pool(x)
        return self.elu(self.fc(x.flatten(1)))


class ParallelCNN(nn.Module):
    def __init__(self, nb=NCH):
        super().__init__()
        self.branches = nn.ModuleList([Branch() for _ in range(nb)])
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(50 * nb, 100), nn.SELU(), nn.Dropout(0.5),
                                  nn.Linear(100, 20), nn.SELU(), nn.Dropout(0.5), nn.Linear(20, 1))

    def forward(self, x):  # x (B,100,18) -> logit (B,)
        feats = [b(x[:, :, i].unsqueeze(1)) for i, b in enumerate(self.branches)]
        return self.head(torch.cat(feats, 1)).squeeze(1)


# ---------------- XAI (all return |attribution| maps of shape (B,100,18)) ----------------
def gradcam_1d(model, x):
    """Per-branch 1D Grad-CAM on each branch's last conv (c4), target = PD logit sign of prediction.
    Branch maps are NOT renormalised per branch, so cross-sensor magnitude is preserved."""
    acts, grads = {}, {}
    hs = []
    for i, b in enumerate(model.branches):
        hs.append(b.c4.register_forward_hook(lambda m, inp, out, i=i: acts.__setitem__(i, out)))
        hs.append(b.c4.register_full_backward_hook(lambda m, gi, go, i=i: grads.__setitem__(i, go[0])))
    x = x.clone().requires_grad_(True)
    logit = model(x)
    sign = torch.where(logit >= 0, 1.0, -1.0)  # explain predicted class
    (logit * sign).sum().backward()
    for h in hs:
        h.remove()
    maps = []
    for i in range(len(model.branches)):
        a, g = acts[i], grads[i]  # (B,16,44)
        w = g.mean(-1, keepdim=True)
        cam = torch.relu((w * a).sum(1, keepdim=True))  # (B,1,44)
        cam = torch.nn.functional.interpolate(cam, size=SEG, mode="linear", align_corners=False)
        maps.append(cam[:, 0])
    return torch.stack(maps, -1).detach().numpy()  # (B,100,18)


def integrated_gradients(f, x, steps=32):
    """f: torch fn x->score (B,). Zero baseline (= no load on any sensor)."""
    base = torch.zeros_like(x)
    tot = torch.zeros_like(x)
    for a in torch.linspace(1 / steps, 1, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        s = f(xi).sum()
        g, = torch.autograd.grad(s, xi)
        tot += g
    return ((x - base) * tot / steps).abs().detach().numpy()


def occlusion(f, x, win=10):
    """Occlude (zero) each (sensor, 10-sample window) cell; attribution = drop in score."""
    with torch.no_grad():
        s0 = f(x)
        out = np.zeros(tuple(x.shape), np.float32)
        for c in range(x.shape[2]):
            for t in range(0, SEG, win):
                xo = x.clone(); xo[:, t:t + win, c] = 0
                d = (s0 - f(xo)).clamp(min=0).numpy()
                out[:, t:t + win, c] = d[:, None]
    return out


# ---------------- clinical ROI: gait events from raw foot-total force ----------------
def gait_event_mask(raw_seg: np.ndarray, half=5, thr_frac=0.1):
    """raw_seg (100,18) in Newtons. Heel strike / toe-off = threshold crossings of each foot's total
    force (threshold = 10% of that foot's max in the segment). ROI = +-5 samples (50 ms) around each
    event, applied to all 9 channels of that foot. Returns bool (100,18)."""
    m = np.zeros((SEG, NCH), bool)
    for tot_idx, chans in ((16, list(range(0, 8)) + [16]), (17, list(range(8, 16)) + [17])):
        f = raw_seg[:, tot_idx]
        if f.max() <= 0:
            continue
        on = f > thr_frac * f.max()
        ev = np.where(np.diff(on.astype(int)) != 0)[0]
        for e in ev:
            m[max(0, e - half):e + half + 1, chans] = True
    return m


def roi_metrics(attr: np.ndarray, roi: np.ndarray, edge=5):
    """attr (100,18) nonneg; returns enrichment on gait events, edge enrichment, top-sensor share,
    pointing hit."""
    a = np.maximum(attr, 0)
    tot = a.sum()
    out = {}
    af = roi.mean()
    out["roi_area_frac"] = float(af)
    out["roi_enrichment"] = float((a[roi].sum() / tot) / af) if tot > 0 and af > 0 else np.nan
    out["pointing_hit"] = float(roi.ravel()[np.argmax(a)]) if tot > 0 else np.nan
    edge_m = np.zeros_like(roi); edge_m[:edge] = True; edge_m[-edge:] = True
    out["edge_enrichment"] = float((a[edge_m].sum() / tot) / edge_m.mean()) if tot > 0 else np.nan
    per_sensor = a.sum(0) / tot if tot > 0 else np.full(NCH, np.nan)
    out["top_sensor_share"] = float(np.nanmax(per_sensor))  # uniform = 1/18 = 0.056
    out["totals_share"] = float(per_sensor[16:].sum())  # uniform = 2/18 = 0.111
    return out, per_sensor
