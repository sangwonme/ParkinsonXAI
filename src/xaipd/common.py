"""Shared helpers: test split listing, preprocessing geometry, stroke masks, metrics."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "parkonix_repo" / "Data" / "drawings"
RESULTS = ROOT / "results"
LABELS = {"healthy": 0, "parkinson": 1}


def list_split(split: str = "testing"):
    """Return list of dicts (path, task, label) for kmader-style split."""
    rows = []
    for task in ("spiral", "wave"):
        for cls, y in LABELS.items():
            for p in sorted((DATA / task / split / cls).glob("*.png")):
                rows.append({"path": p, "task": task, "label": y, "name": p.name})
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---- geometry: A1 uses ImageOps.fit (center crop to square, then resize) ----
def fit_box(w: int, h: int, size=(224, 224)):
    """Crop box (in original coords) used by ImageOps.fit with centering 0.5."""
    tw, th = size
    target_ratio = tw / th
    if w / h > target_ratio:  # too wide
        cw, ch = round(h * target_ratio), h
    else:
        cw, ch = w, round(w / target_ratio)
    left = (w - cw) / 2
    top = (h - ch) / 2
    return int(round(left)), int(round(top)), int(round(left)) + cw, int(round(top)) + ch


def a1_input(img: Image.Image) -> np.ndarray:
    im = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    return (np.asarray(im).astype(np.float32) / 127.5) - 1.0


def cam_to_original(cam: np.ndarray, orig_wh, mode: str) -> np.ndarray:
    """Map a CAM defined on the model input frame back to the original image frame.
    mode='fit'  -> A1 (center-crop); outside crop = 0 (model never saw it)
    mode='resize' -> A3 (squash resize)."""
    w, h = orig_wh
    c = Image.fromarray(cam.astype(np.float32), mode="F")
    if mode == "resize":
        return np.asarray(c.resize((w, h), Image.Resampling.BILINEAR))
    l, t, r, b = fit_box(w, h)
    out = np.zeros((h, w), np.float32)
    out[t:b, l:r] = np.asarray(c.resize((r - l, b - t), Image.Resampling.BILINEAR))
    return out


# ---- stroke mask ----
def stroke_mask(img: Image.Image, dilate: int = 3) -> np.ndarray:
    """Ink pixels via Otsu on grayscale (ink is darker than paper), then dilation."""
    import cv2

    g = np.asarray(img.convert("L"))
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # remove tiny specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            keep[lab == i] = 255
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        keep = cv2.dilate(keep, k)
    return keep > 0


# ---- metrics ----
def norm01(cam):
    cam = np.maximum(cam, 0)
    mx = cam.max()
    return cam / mx if mx > 0 else cam


def stroke_metrics(cam: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None):
    """cam, mask in original frame. valid = region the model actually saw."""
    if valid is None:
        valid = np.ones_like(mask, bool)
    c = np.maximum(cam, 0) * valid
    tot = c.sum()
    m = mask & valid
    area_frac = m.sum() / max(valid.sum(), 1)
    if tot <= 0:
        return dict(energy_on_stroke=np.nan, bg_energy=np.nan, area_frac=area_frac,
                    enrichment=np.nan, pointing_hit=np.nan, top20_on_stroke=np.nan)
    e = c[m].sum() / tot
    yx = np.unravel_index(np.argmax(c), c.shape)
    thr = np.quantile(c[valid], 0.8)
    top = (c >= thr) & valid
    return dict(
        energy_on_stroke=float(e),
        bg_energy=float(1 - e),
        area_frac=float(area_frac),
        enrichment=float(e / area_frac) if area_frac > 0 else np.nan,
        pointing_hit=float(m[yx]),
        top20_on_stroke=float((top & m).sum() / max(top.sum(), 1)),
    )


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).correlation)


def topk_iou(a, b, q=0.8, valid=None):
    if valid is None:
        valid = np.ones_like(a, bool)
    ta = (a >= np.quantile(a[valid], q)) & valid
    tb = (b >= np.quantile(b[valid], q)) & valid
    u = (ta | tb).sum()
    return float((ta & tb).sum() / u) if u else np.nan


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bootstrap_acc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    accs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        accs.append((y[i] == p[i]).mean())
    return float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))
