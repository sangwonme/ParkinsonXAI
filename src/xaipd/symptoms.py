"""Symptom-operator model (interpretable by design) for PD gait severity on CARE-PD.

    y_hat(x) = b + sum_k e_k(x) + r(x),   e_k = g_k(c_k(x; theta_k)) - g_k(c_k_ref)

c_k : differentiable symptom operators on 3D joint trajectories (COCO-13 from SMPL FK, 30 fps).
      Invalid frames (all joints == 0, e.g. zero padding) are masked; every length is divided by the
      walker's leg length -> padding and body-size shortcuts are removed by construction.
g_k : monotone shape functions (sum of positive-weight sigmoids; direction fixed by clinical prior,
      or learned sign when the prior is ambiguous) -> monotone by construction (no penalty needed).
r   : small residual encoder, penalised by lambda * E[r^2].
Output: cumulative-link ordinal model  P(y > j) = sigmoid(y_hat - tau_j), classes {0, 1, >=2}.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT

FPS = 30.0
NFFT = 512
CACHE = ROOT / "data" / "processed" / "carepd_coco13.npz"
J_NOSE, J_LSH, J_RSH, J_LEL, J_REL, J_LWR, J_RWR, J_LHIP, J_RHIP, J_LKN, J_RKN, J_LAN, J_RAN = range(13)
SYMPTOMS = ["stride_length", "cadence", "arm_swing", "arm_asymmetry", "rhythm_irregularity",
            "trunk_flexion", "walking_speed", "freeze_index"]
NICE = {"stride_length": "stride length", "cadence": "cadence", "arm_swing": "arm swing",
        "arm_asymmetry": "arm swing asymmetry", "rhythm_irregularity": "rhythm irregularity",
        "trunk_flexion": "trunk flexion", "walking_speed": "walking speed", "freeze_index": "freezing index"}
# clinical prior direction of severity: -1 = smaller value -> more severe, +1 = larger -> more severe, 0 = learn
PRIOR = {"stride_length": -1, "cadence": 0, "arm_swing": -1, "arm_asymmetry": +1, "rhythm_irregularity": +1,
         "trunk_flexion": +1, "walking_speed": -1, "freeze_index": +1}


# ----------------------------------------------------------------------------- data
def build_cache():
    """All UPDRS-labelled CARE-PD walks -> world COCO-13 joints (30 fps). Stored without pickle."""
    from . import care_pd as CP
    J, nose = CP.smpl_rest()
    Xs, y, yr, ds, subj, walk = [], [], [], [], [], []
    for d in CP.UPDRS_SETS:
        for w in CP.iter_walks(d):
            if w["updrs"] is None:
                continue
            X3 = CP.fk_coco13(w["pose"], w["trans"], J, nose)
            fps = w["fps"] or 30
            if abs(fps - FPS) > 1:
                T = X3.shape[0]; tn = np.arange(0, T, fps / FPS)
                X3 = np.stack([np.stack([np.interp(tn, np.arange(T), X3[:, j, c]) for c in range(3)], -1) for j in range(13)], 1)
            Xs.append(X3.astype(np.float32)); y.append(min(int(w["updrs"]), 2)); yr.append(int(w["updrs"]))
            ds.append(d); subj.append(w["subj"]); walk.append(w["walk"])
    off = np.cumsum([0] + [len(x) for x in Xs])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=np.concatenate(Xs), off=off, y=np.array(y), y_raw=np.array(yr),
                        ds=np.array(ds), subj=np.array(subj), walk=np.array(walk))


def load_cache():
    d = np.load(CACHE, allow_pickle=False)
    X, off = d["X"], d["off"]
    walks = [X[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    return walks, {k: d[k] for k in ("y", "y_raw", "ds", "subj", "walk")}


def batchify(walks, idx, device="cpu"):
    T = max(len(walks[i]) for i in idx)
    B = np.zeros((len(idx), T, 13, 3), np.float32)
    for b, i in enumerate(idx):
        B[b, :len(walks[i])] = walks[i]
    return torch.from_numpy(B).to(device)


# ----------------------------------------------------------------------------- injections (on world joints)
def _scale_rel(X, joints, anchor, alpha):
    X = X.copy()
    for j in joints:
        rel = X[:, j] - X[:, anchor]
        m = rel.mean(0, keepdims=True)
        X[:, j] = X[:, anchor] + m + alpha * (rel - m)
    return X


def forward_axis(X):
    mh = (X[:, J_LHIP] + X[:, J_RHIP]) / 2
    v = mh[-1] - mh[0]; v[1] = 0
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0, 0, 1.0])


def inject(X, kind, alpha=None):
    """Synthetic, clinically motivated perturbations of a healthy walk."""
    if kind == "reduced_arm_swing":
        a = 0.3 if alpha is None else alpha
        X = _scale_rel(X, [J_LEL, J_LWR], J_LSH, a)
        return _scale_rel(X, [J_REL, J_RWR], J_RSH, a)
    if kind == "arm_asymmetry":  # one arm (left) nearly stops swinging
        return _scale_rel(X, [J_LEL, J_LWR], J_LSH, 0.15 if alpha is None else alpha)
    if kind == "short_stride":
        a = 0.55 if alpha is None else alpha
        X = _scale_rel(X, [J_LKN, J_LAN], J_LHIP, a)
        X = _scale_rel(X, [J_RKN, J_RAN], J_RHIP, a)
        f = forward_axis(X)
        mh = (X[:, J_LHIP] + X[:, J_RHIP]) / 2
        prog = (mh - mh[:1]) @ f  # forward progression
        return X + ((a - 1.0) * prog)[:, None, None] * f[None, None, :]
    raise ValueError(kind)


INJ_TARGET = {"reduced_arm_swing": {"arm_swing"}, "arm_asymmetry": {"arm_asymmetry"},
              "short_stride": {"stride_length", "walking_speed"}}


# ----------------------------------------------------------------------------- operators
def _masked_std(v, m):  # v (B,T), m (B,T)
    n = m.sum(1).clamp(min=1)
    mu = (v * m).sum(1) / n
    return torch.sqrt(((v - mu[:, None]) ** 2 * m).sum(1) / n + 1e-8)


def _mean(v, m):
    return (v * m).sum(1) / m.sum(1).clamp(min=1)


class SymptomOperators(nn.Module):
    """x: (B,T,13,3) world joints (zeros = invalid frame). Returns (B,K) symptom values."""

    def __init__(self):
        super().__init__()
        f = torch.fft.rfftfreq(NFFT, 1 / FPS)
        self.register_buffer("freqs", f)
        # learnable: gait band for cadence/rhythm (Hz), freeze bands (Hz), arm joint mix, spectral temperature
        self.gait_lo = nn.Parameter(torch.tensor(0.4)); self.gait_hi = nn.Parameter(torch.tensor(1.6))
        self.frz_lo = nn.Parameter(torch.tensor(3.0)); self.frz_hi = nn.Parameter(torch.tensor(8.0))
        self.loco_hi = nn.Parameter(torch.tensor(3.0))
        self.arm_mix = nn.Parameter(torch.tensor(1.5))  # sigmoid -> weight of wrist vs elbow
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.05)))

    def band(self, lo, hi, sharp=4.0):
        return torch.sigmoid(sharp * (self.freqs - lo)) * torch.sigmoid(sharp * (hi - self.freqs))

    def spectrum(self, v, m):
        v = (v - _mean(v, m)[:, None]) * m
        first = m.argmax(1)  # align every sequence to its first valid frame (padding-invariant)
        v = torch.stack([torch.roll(v[b], -int(first[b])) for b in range(v.shape[0])])
        n_valid = int(m.sum(1).max())
        v = v[:, :max(n_valid, 2)]
        T = v.shape[1]
        if T > NFFT:  # use the central NFFT frames
            s = (T - NFFT) // 2; v = v[:, s:s + NFFT]
        return torch.fft.rfft(v, n=NFFT).abs() ** 2

    def forward(self, x):
        valid = (x.abs().sum(dim=(2, 3)) > 0).float()  # (B,T)
        hip = (x[:, :, J_LHIP] + x[:, :, J_RHIP]) / 2
        sh = (x[:, :, J_LSH] + x[:, :, J_RSH]) / 2
        # body size: leg length (hip-knee + knee-ankle), median over valid frames
        leg = (torch.linalg.norm(x[:, :, J_LHIP] - x[:, :, J_LKN], dim=-1) + torch.linalg.norm(x[:, :, J_LKN] - x[:, :, J_LAN], dim=-1)
               + torch.linalg.norm(x[:, :, J_RHIP] - x[:, :, J_RKN], dim=-1) + torch.linalg.norm(x[:, :, J_RKN] - x[:, :, J_RAN], dim=-1)) / 2
        L = torch.stack([torch.median(leg[b][valid[b] > 0]) for b in range(x.shape[0])]).clamp(min=1e-3)
        # forward axis from pelvis displacement (horizontal)
        idx_first = valid.argmax(1); idx_last = valid.shape[1] - 1 - valid.flip(1).argmax(1)
        bi = torch.arange(x.shape[0], device=x.device)
        disp = hip[bi, idx_last] - hip[bi, idx_first]
        disp = disp * torch.tensor([1.0, 0.0, 1.0], device=x.device)
        dur = ((idx_last - idx_first).float() / FPS).clamp(min=1e-3)
        fwd = F.normalize(disp + torch.tensor([0, 0, 1e-6], device=x.device), dim=-1)
        up = torch.tensor([0.0, 1.0, 0.0], device=x.device).expand_as(fwd)
        proj = lambda v: (v * fwd[:, None, :]).sum(-1)

        # 1 stride: ankle separation along the walking direction (per leg length), sinusoid peak-to-peak
        d = proj(x[:, :, J_LAN] - x[:, :, J_RAN]) / L[:, None]
        stride = 2 * math.sqrt(2) * _masked_std(d, valid)
        # 2 cadence & 5 rhythm irregularity from the spectrum of d inside a learnable gait band
        P = self.spectrum(d, valid) * self.band(self.gait_lo, self.gait_hi)
        w = torch.softmax(torch.log(P + 1e-10) / self.log_temp.exp().clamp(min=1e-3) * 0.05, dim=1)
        f_peak = (w * self.freqs).sum(1)
        cadence = 2 * f_peak * 60.0  # steps per minute
        Pn = P / P.sum(1, keepdim=True).clamp(min=1e-10)
        rhythm = -(Pn * torch.log(Pn + 1e-10)).sum(1) / math.log(P.shape[1])  # normalised spectral entropy
        # 3 arm swing amplitude & 4 asymmetry: (mixed wrist/elbow) forward excursion relative to shoulder
        a = torch.sigmoid(self.arm_mix)
        armL = a * x[:, :, J_LWR] + (1 - a) * x[:, :, J_LEL] - x[:, :, J_LSH]
        armR = a * x[:, :, J_RWR] + (1 - a) * x[:, :, J_REL] - x[:, :, J_RSH]
        AL = 2 * math.sqrt(2) * _masked_std(proj(armL) / L[:, None], valid)
        AR = 2 * math.sqrt(2) * _masked_std(proj(armR) / L[:, None], valid)
        arm = (AL + AR) / 2
        asym = (AL - AR).abs() / (AL + AR + 1e-3)
        # 6 trunk flexion: forward lean angle of the hip->shoulder vector (radians)
        tv = sh - hip
        lean = torch.atan2(proj(tv), (tv * up[:, None, :]).sum(-1))
        trunk = _mean(lean, valid)
        # 7 walking speed: pelvis progression per second per leg length
        speed = (disp.norm(dim=-1) / dur) / L
        # 8 freezing index: log power ratio of vertical ankle velocity, freeze band vs locomotion band
        vy = torch.diff(x[:, :, J_LAN, 1] + x[:, :, J_RAN, 1], dim=1, prepend=x[:, :1, J_LAN, 1] + x[:, :1, J_RAN, 1]) * FPS / L[:, None]
        mv = valid * torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], 1)
        Pv = self.spectrum(vy, mv)
        fr = (Pv * self.band(self.frz_lo, self.frz_hi)).sum(1); lo = (Pv * self.band(torch.tensor(0.5, device=x.device), self.loco_hi)).sum(1)
        freeze = torch.log((fr + 1e-6) / (lo + 1e-6))
        return torch.stack([stride, cadence, arm, asym, rhythm, trunk, speed, freeze], 1)


class MonotoneShape(nn.Module):
    """g(z) = sign * sum_j softplus(a_j) * sigmoid(softplus(w_j) * (z - b_j)); monotone in z."""

    def __init__(self, prior, n=6):
        super().__init__()
        self.a = nn.Parameter(torch.full((n,), -1.0)); self.w = nn.Parameter(torch.zeros(n))
        self.b = nn.Parameter(torch.linspace(-2, 2, n))
        self.prior = prior
        self.sign_raw = nn.Parameter(torch.tensor(0.0)) if prior == 0 else None

    def sign(self):
        return torch.tanh(self.sign_raw * 3) if self.prior == 0 else torch.tensor(float(self.prior))

    def forward(self, z):  # z (B,)
        s = (F.softplus(self.a)[None] * torch.sigmoid(F.softplus(self.w)[None] * 2 * (z[:, None] - self.b[None]))).sum(1)
        return self.sign().to(z.device) * s


class Residual(nn.Module):
    """Capacity-limited residual encoder on body-size-normalised, pelvis-centred joints (masked pooling)."""

    def __init__(self, ch=8):
        super().__init__()
        self.c1 = nn.Conv1d(39, ch, 9, padding=4); self.c2 = nn.Conv1d(ch, ch, 9, padding=4, stride=2)
        self.out = nn.Linear(ch, 1)

    def forward(self, x, valid, L):
        hip = (x[:, :, J_LHIP] + x[:, :, J_RHIP]) / 2
        z = ((x - hip[:, :, None]) / L[:, None, None, None]) * valid[:, :, None, None]
        z = z.flatten(2).transpose(1, 2)  # (B,39,T)
        h = F.relu(self.c1(z)); h = F.relu(self.c2(h))
        m = valid[:, ::2][:, :h.shape[2]]
        h = (h * m[:, None]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)
        return self.out(h)[:, 0]


class SymptomModel(nn.Module):
    def __init__(self, use_residual=True, ncls=3):
        super().__init__()
        self.ops = SymptomOperators()
        self.g = nn.ModuleList([MonotoneShape(PRIOR[k]) for k in SYMPTOMS])
        self.res = Residual() if use_residual else None
        self.register_buffer("mu", torch.zeros(len(SYMPTOMS))); self.register_buffer("sd", torch.ones(len(SYMPTOMS)))
        self.register_buffer("ref", torch.zeros(len(SYMPTOMS)))  # healthy reference (standardised)
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.tau_raw = nn.Parameter(torch.tensor([0.0, 0.0][: ncls - 1]))

    def taus(self):
        t = [self.tau_raw[0]]
        for j in range(1, len(self.tau_raw)):
            t.append(t[-1] + F.softplus(self.tau_raw[j]) + 0.1)
        return torch.stack(t)

    def forward(self, x):
        c = self.ops(x)
        z = (c - self.mu) / self.sd
        e = torch.stack([g(z[:, k]) - g(self.ref[k:k + 1].expand(len(z))) for k, g in enumerate(self.g)], 1)
        r = torch.zeros(len(x), device=x.device)
        if self.res is not None:
            valid = (x.abs().sum(dim=(2, 3)) > 0).float()
            hipk = (x[:, :, J_LHIP] - x[:, :, J_LKN]).norm(dim=-1) + (x[:, :, J_LKN] - x[:, :, J_LAN]).norm(dim=-1)
            L = torch.stack([torch.median(hipk[b][valid[b] > 0]) for b in range(len(x))]).clamp(min=1e-3)
            r = self.res(x, valid, L)
        yhat = self.bias + e.sum(1) + r
        return dict(c=c, z=z, e=e, r=r, yhat=yhat)

    def class_probs(self, yhat):
        cum = torch.sigmoid(yhat[:, None] - self.taus()[None])  # P(y > j)
        p = torch.cat([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1)
        return p.clamp(min=1e-6)
