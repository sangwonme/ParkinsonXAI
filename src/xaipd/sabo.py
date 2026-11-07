"""Sabo et al. 2021 ST-GCN (TaatiTeam/stgcn_parkinsonism_prediction) — minimal faithful port so the
published checkpoints load with torch.load(weights_only=True) and strict key matching.
Architecture: ST_GCN_18_ordinal_smaller_2_position_pretrain, layout coco_simplified_head (13 joints),
spatial strategy (K=3), temporal kernel 13, 4 st-gcn blocks (32,32,64s2,64), stage-2 regression head.
Preprocessing (from configs + loader_tri.py): 13 joints [Nose, LSh, RSh, LEl, REl, LWr, RWr, LHip, RHip,
LKnee, RKnee, LAnk, RAnk]; NaN -> per-column mean; x/1920-0.5, y/1080-0.5; joints with conf==0 -> 0;
zero-pad at the beginning to 120 frames; crop to 120 frames (random in the original eval; here we
average all 120-frame windows with stride 15, documented deviation); prediction = round & clip [0, 2]."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT

SABO = ROOT / "models" / "raw" / "sabo_stgcn"
ZOO = SABO / "sample_data" / "model_zoo"
DATA = SABO / "sample_data" / "data"
JOINTS = ["Nose", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
          "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"]
T_IN = 120
RES = (1920.0, 1080.0)


class GCN(nn.Module):  # ConvTemporalGraphical (gconv_origin), t_kernel=1
    def __init__(self, cin, cout, K):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(cin, cout * K, kernel_size=(1, 1))

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.K, kc // self.K, t, v)
        return torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()


class Block(nn.Module):
    def __init__(self, cin, cout, K, tk, stride=1, residual=True):
        super().__init__()
        self.gcn = GCN(cin, cout, K)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(), nn.Conv2d(cout, cout, (tk, 1), (stride, 1), ((tk - 1) // 2, 0)),
                                 nn.BatchNorm2d(cout), nn.Dropout(0.0))
        if not residual:
            self.res_kind = "none"
        elif cin == cout and stride == 1:
            self.res_kind = "id"
        else:
            self.res_kind = "conv"
            self.residual = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x, A):
        res = 0 if self.res_kind == "none" else (x if self.res_kind == "id" else self.residual(x))
        return F.relu(self.tcn(self.gcn(x, A)) + res)


class Encoder(nn.Module):
    def __init__(self, cin=2, V=13, K=3, tk=13):
        super().__init__()
        self.register_buffer("A", torch.zeros(K, V, V))
        self.data_bn = nn.BatchNorm1d(cin * V)
        self.st_gcn_networks = nn.ModuleList([Block(cin, 32, K, tk, 1, residual=False), Block(32, 32, K, tk),
                                              Block(32, 64, K, tk, 2), Block(64, 64, K, tk)])
        self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(K, V, V)) for _ in range(4)])
        self.feat = None  # last block output, for Grad-CAM

    def forward(self, x):  # x (N, C, T, V)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).reshape(N, V * C, T)
        x = self.data_bn(x).view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for blk, imp in zip(self.st_gcn_networks, self.edge_importance):
            x = blk(x, self.A * imp)
        self.feat = x  # (N, 64, T/2, V)
        return F.avg_pool2d(x, x.size()[2:])  # (N,64,1,1)


class SaboSTGCN(nn.Module):
    def __init__(self, cin=2):
        super().__init__()
        self.encoder = Encoder(cin)
        self.pretrain_head = nn.Conv2d(64, 8, 1)
        self.classification_head = nn.Conv2d(64, 1, 1)
        self.head = nn.Conv2d(64, 1, 1)  # saved alias of classification_head in stage 2

    def forward(self, x):  # (N, C, T, V) -> raw score (N,)
        return self.head(self.encoder(x)).view(-1)


def checkpoints(kind="detectron"):
    sub = {"detectron": "detectron_models_to_share/v2/13/0.2", "kinect2d": "kinect2d_models_to_share/v2/13/0.0"}[kind]
    return sorted((ZOO / sub).glob("*/checkpoint_final.pt"), key=lambda p: int(p.parent.name))


def load(path: Path, cin=2):
    sd = torch.load(path, map_location="cpu", weights_only=True)  # NEVER weights_only=False
    m = SaboSTGCN(cin)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert torch.allclose(sd["head.weight"], sd["classification_head.weight"]), "head != classification_head"
    return m.eval()


# ------------------------------------------------------------- input pipeline
def preprocess_xyconf(xy: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """xy (T,13,2) in the 'normalized_100_center' pixel frame; returns (2,T,13) model input before padding."""
    x = xy.astype(np.float32).copy()
    x[..., 0] = x[..., 0] / RES[0] - 0.5
    x[..., 1] = x[..., 1] / RES[1] - 0.5
    if conf is not None:
        x[conf == 0] = 0.0
    return x.transpose(2, 0, 1)


def load_csv(path: Path):
    df = pd.read_csv(path)
    if "Nose_x" not in df and "Head_x" in df:  # Kinect CSVs: head joint stands in for the nose
        df["Nose_x"], df["Nose_y"] = df["Head_x"], df["Head_y"]
    dfi = df.fillna(df.mean(numeric_only=True))
    xy = np.stack([np.stack([dfi[f"{j}_x"].values, dfi[f"{j}_y"].values], -1) for j in JOINTS], 1)
    conf = np.stack([dfi[f"{j}_conf"].values for j in JOINTS], 1) if f"{JOINTS[0]}_conf" in df else None
    if conf is not None:
        conf = np.nan_to_num(conf, nan=0.0)
    return dict(x=preprocess_xyconf(xy, conf), updrs=int(df["UPDRS_gait"].iloc[0]), sas=df["SAS_gait"].iloc[0],
                name=path.stem, T=len(df))


def windows(x: np.ndarray, stride=15):
    """x (C,T,V) -> list of (C,120,V) windows (zero-padded at beginning if short) and pad length."""
    C, T, V = x.shape
    if T <= T_IN:
        out = np.zeros((C, T_IN, V), np.float32); out[:, T_IN - T:] = x
        return [out], T_IN - T
    starts = list(range(0, T - T_IN + 1, stride))
    if starts[-1] != T - T_IN:
        starts.append(T - T_IN)
    return [x[:, s:s + T_IN] for s in starts], 0


def predict_walk(model, x: np.ndarray):
    ws, _ = windows(x)
    with torch.no_grad():
        raw = model(torch.from_numpy(np.stack(ws))).numpy()
    r = float(raw.mean())
    return r, int(np.clip(np.round(r), 0, 2))
