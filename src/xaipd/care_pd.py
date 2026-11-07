"""CARE-PD (canonicalised SMPL gait) -> 3D COCO-13 joints -> 2D projections for the Sabo ST-GCN.
SMPL forward kinematics only (rest joints from SMPL_NEUTRAL J_regressor @ v_template, betas = 0 as in
CARE-PD's anonymisation). Nose = SMPL vertex 332 rigidly attached to the head joint (15).
Canonical frame (CARE-PD card): x lateral, y up, z forward (walking direction)."""
from __future__ import annotations

import numpy as np

from .common import ROOT
from .safe_pickle import safe_load

CDIR = ROOT / "data" / "raw" / "care_pd"
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
# COCO-13 order used by Sabo: Nose, LSh, RSh, LEl, REl, LWr, RWr, LHip, RHip, LKnee, RKnee, LAnk, RAnk
SMPL_IDX = [None, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
NOSE_VERT, HEAD_J = 332, 15
UPDRS_SETS = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]


def smpl_rest():
    """Returns (J_rest (24,3), nose_rest (3,)). Loads SMPL_NEUTRAL.pkl with the restricted unpickler."""
    from .safe_pickle import safe_load_smpl
    m = safe_load_smpl(ROOT / "models" / "raw" / "smpl" / "SMPL_NEUTRAL.pkl")
    v = np.asarray(m["v_template"], np.float64)
    Jr = m["J_regressor"]
    Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.asarray(Jr)
    return Jr @ v, v[NOSE_VERT]


def rodrigues(aa):  # (...,3) -> (...,3,3)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def fk_coco13(pose, trans, J, nose_rest):
    """pose (T,72) axis-angle, trans (T,3) -> (T,13,3) world joints."""
    T = pose.shape[0]
    R = rodrigues(pose.reshape(T, 24, 3))
    Rg = np.zeros((T, 24, 3, 3)); P = np.zeros((T, 24, 3))
    for i, p in enumerate(PARENTS):
        if p < 0:
            Rg[:, i] = R[:, i]; P[:, i] = J[i]
        else:
            Rg[:, i] = Rg[:, p] @ R[:, i]
            P[:, i] = P[:, p] + np.einsum("tij,j->ti", Rg[:, p], J[i] - J[p])
    nose = P[:, HEAD_J] + np.einsum("tij,j->ti", Rg[:, HEAD_J], nose_rest - J[HEAD_J])
    out = np.stack([nose] + [P[:, j] for j in SMPL_IDX[1:]], 1)
    return out + trans[:, None, :]


VIEWS = {  # orthographic image axes (u right, v down) expressed in canonical (x lat, y up, z fwd)
    # SMPL left joints have +x. Camera in front (looking along -z): subject's left appears on image
    # right -> u = +x. This matches the Sabo detectron samples (LShoulder_x > mid-hip x = 100).
    "frontal": lambda X: np.stack([X[..., 0], -X[..., 1]], -1),    # walking toward camera
    "back": lambda X: np.stack([-X[..., 0], -X[..., 1]], -1),      # walking away from camera
    "side": lambda X: np.stack([X[..., 2], -X[..., 1]], -1),       # sagittal view
    "oblique45": lambda X: np.stack([(X[..., 0] + X[..., 2]) / np.sqrt(2), -X[..., 1]], -1),
}


def to_sabo_frame(xy, torso=190.0):
    """Per-frame mid-hip -> (100,100) (matches every Sabo sample CSV: mid-hip exactly 100,100);
    one global scale per walk so the median shoulder-mid to hip-mid length = `torso` units
    (Sabo detectron samples: 153-217, median ~190)."""
    mh = (xy[:, 7] + xy[:, 8]) / 2
    ms = (xy[:, 1] + xy[:, 2]) / 2
    s = torso / np.median(np.linalg.norm(ms - mh, axis=-1))
    return (xy - mh[:, None]) * s + 100.0


def load_set(name):
    return safe_load(CDIR / f"{name}_canonical.pkl")


def iter_walks(name):
    d = load_set(name)
    for subj, walks in d.items():
        for w, v in walks.items():
            yield dict(dataset=name, subj=f"{name}:{subj}", walk=w, pose=v["pose"], trans=v["trans"],
                       fps=v.get("fps", 30), updrs=v.get("UPDRS_GAIT"))
