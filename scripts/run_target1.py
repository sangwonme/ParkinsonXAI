"""Target 1 (spiral/wave): E0 accuracy, Grad-CAM/++/Score-CAM, E2 stroke attention, E4 A1-vs-A3
consistency, plus a cheap E7 stroke-only shortcut check. No training.
Run: .venv/Scripts/python.exe scripts/run_target1.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from xaipd import common as C

OUT = C.RESULTS / "target1"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["gradcam", "gradcampp", "scorecam"]


def stroke_only(img: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(img).copy()
    a[~mask] = 255
    return Image.fromarray(a)


# ---------------- A1 (Keras, CPU) ----------------
def run_a1(rows):
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from xaipd import a1_keras as A1

    model = A1.load()
    cam_model, conv_name, head = A1.build_cam_model(model)
    print("A1 target conv:", conv_name, "| head:", head)
    res = {}
    for r in rows:
        img = C.load_rgb(r["path"])
        x = C.a1_input(img)
        p_ref = model.predict(x[None], verbose=0)[0]
        _, p_cm = cam_model(x[None], training=False)
        p_cm = p_cm.numpy()[0]
        assert np.allclose(p_ref, p_cm, atol=1e-4), (p_ref, p_cm)
        pred = int(np.argmax(p_ref))
        mask = C.stroke_mask(img)
        xs = C.a1_input(stroke_only(img, C.stroke_mask(img, dilate=1)))
        p_so = model.predict(xs[None], verbose=0)[0]
        cams = {
            "gradcam": A1.gradcam(cam_model, x, pred),
            "gradcampp": A1.gradcam(cam_model, x, pred, plus=True),
            "scorecam": A1.scorecam(cam_model, x, pred),
        }
        res[r["name"]] = dict(prob_pd=float(p_ref[1]), pred=pred, prob_pd_strokeonly=float(p_so[1]),
                              cams=cams, mode="fit")
    return res, conv_name


# ---------------- A3 (Swin, torch GPU) ----------------
def run_a3(rows):
    import torch
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from xaipd import a3_swin as A3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = A3.load(dev)
    tl = A3.target_layers(model)
    cams_obj = {
        "gradcam": GradCAM(model=model, target_layers=tl, reshape_transform=A3.reshape_transform),
        "gradcampp": GradCAMPlusPlus(model=model, target_layers=tl, reshape_transform=A3.reshape_transform),
        "scorecam": ScoreCAM(model=model, target_layers=tl, reshape_transform=A3.reshape_transform),
    }
    cams_obj["scorecam"].batch_size = 64
    res = {}
    for r in rows:
        img = C.load_rgb(r["path"])
        x = A3.preprocess(img)[None].to(dev)
        with torch.no_grad():
            p = torch.softmax(model(x), -1)[0].cpu().numpy()
            xs = A3.preprocess(stroke_only(img, C.stroke_mask(img, dilate=1)))[None].to(dev)
            p_so = torch.softmax(model(xs), -1)[0].cpu().numpy()
        pred = int(np.argmax(p))
        cams = {}
        for m, obj in cams_obj.items():
            cams[m] = obj(input_tensor=x, targets=[ClassifierOutputTarget(pred)], aug_smooth=False,
                          eigen_smooth=False)[0]  # 224x224, already upsampled+normalized
        res[r["name"]] = dict(prob_pd=float(p[1]), pred=pred, prob_pd_strokeonly=float(p_so[1]),
                              cams=cams, mode="resize")
    return res


def summarize_acc(rows, res, tag):
    out = {}
    for task in ("spiral", "wave", "all"):
        rr = [r for r in rows if task == "all" or r["task"] == task]
        y = np.array([r["label"] for r in rr])
        p = np.array([res[r["name"]]["pred"] for r in rr])
        pso = np.array([int(res[r["name"]]["prob_pd_strokeonly"] >= 0.5) for r in rr])
        tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        lo, hi = C.bootstrap_acc(y, p)
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, [res[r["name"]]["prob_pd"] for r in rr]))
        except Exception:
            auc = float("nan")
        out[task] = dict(n=len(y), acc=float((p == y).mean()), ci95=[lo, hi], auc=auc,
                         sens=tp / max((y == 1).sum(), 1), spec=tn / max((y == 0).sum(), 1),
                         acc_strokeonly=float((pso == y).mean()))
    print(tag, json.dumps(out, indent=1))
    return out


def overlay(ax, img, cam, title):
    ax.imshow(img)
    if cam is not None:
        ax.imshow(C.norm01(cam), cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    rows = C.list_split("testing")
    assert len(rows) == 60, len(rows)
    manifest = [dict(name=r["name"], task=r["task"], label=r["label"], sha256=C.sha256(r["path"])) for r in rows]
    C.write_csv(OUT / "test_manifest.csv", manifest)

    which = sys.argv[1:] or ["a1", "a3"]
    allres = {}
    if "a1" in which:
        allres["A1_MobileNet_TM"], conv = run_a1(rows)
    if "a3" in which:
        allres["A3_SwinTiny"] = run_a3(rows)

    acc = {k: summarize_acc(rows, v, k) for k, v in allres.items()}
    json.dump(acc, open(OUT / "e0_accuracy.json", "w"), indent=1)

    # per-image predictions + E2 metrics (original image frame)
    per_img, cams_orig = [], {}
    for k, res in allres.items():
        for r in rows:
            img = C.load_rgb(r["path"])
            w, h = img.size
            mask = C.stroke_mask(img)
            if res[r["name"]]["mode"] == "fit":
                l, t, rr_, b = C.fit_box(w, h)
                valid = np.zeros((h, w), bool); valid[t:b, l:rr_] = True
            else:
                valid = np.ones((h, w), bool)
            row = dict(model=k, name=r["name"], task=r["task"], label=r["label"],
                       prob_pd=res[r["name"]]["prob_pd"], pred=res[r["name"]]["pred"],
                       prob_pd_strokeonly=res[r["name"]]["prob_pd_strokeonly"])
            for m in METHODS:
                co = C.cam_to_original(res[r["name"]]["cams"][m], (w, h), res[r["name"]]["mode"])
                cams_orig[(k, r["name"], m)] = (co, valid)
                for kk, vv in C.stroke_metrics(co, mask, valid).items():
                    row[f"{m}_{kk}"] = vv
            per_img.append(row)
    C.write_csv(OUT / "per_image.csv", per_img)

    # E2 summary
    e2 = {}
    for k in allres:
        for task in ("spiral", "wave"):
            rr = [x for x in per_img if x["model"] == k and x["task"] == task]
            for m in METHODS:
                e2[f"{k}|{task}|{m}"] = {
                    s: float(np.nanmean([x[f"{m}_{s}"] for x in rr]))
                    for s in ("bg_energy", "area_frac", "enrichment", "pointing_hit", "top20_on_stroke")
                }
    json.dump(e2, open(OUT / "e2_stroke_attention.json", "w"), indent=1)

    # E4 consistency: A1 vs A3 (same method) and method-vs-method within model
    e4 = []
    names = [r["name"] for r in rows]
    tasks = {r["name"]: r["task"] for r in rows}
    if len(allres) == 2:
        for n in names:
            for m in METHODS:
                a, va = cams_orig[("A1_MobileNet_TM", n, m)]
                b, _ = cams_orig[("A3_SwinTiny", n, m)]
                e4.append(dict(pair=f"A1-vs-A3|{m}", name=n, task=tasks[n],
                               spearman=C.spearman(a[va], b[va]), top20_iou=C.topk_iou(a, b, valid=va)))
    for k in allres:
        for n in names:
            for m1, m2 in (("gradcam", "gradcampp"), ("gradcam", "scorecam"), ("gradcampp", "scorecam")):
                a, va = cams_orig[(k, n, m1)]
                b, _ = cams_orig[(k, n, m2)]
                e4.append(dict(pair=f"{k}|{m1}-vs-{m2}", name=n, task=tasks[n],
                               spearman=C.spearman(a[va], b[va]), top20_iou=C.topk_iou(a, b, valid=va)))
    C.write_csv(OUT / "e4_consistency_per_image.csv", e4)
    e4s = {}
    for pair in sorted({x["pair"] for x in e4}):
        for task in ("spiral", "wave"):
            rr = [x for x in e4 if x["pair"] == pair and x["task"] == task]
            e4s[f"{pair}|{task}"] = dict(spearman=float(np.nanmean([x["spearman"] for x in rr])),
                                         top20_iou=float(np.nanmean([x["top20_iou"] for x in rr])))
    json.dump(e4s, open(OUT / "e4_consistency.json", "w"), indent=1)
    print(json.dumps(e2, indent=1)); print(json.dumps(e4s, indent=1))

    # Figures: per model, 8 samples x (image + 3 methods); and A1 vs A3 side-by-side
    fig_dir = OUT / "figures"; fig_dir.mkdir(exist_ok=True)
    sel = []
    for task in ("spiral", "wave"):
        for y in (0, 1):
            sel += [r for r in rows if r["task"] == task and r["label"] == y][:2]
    for k in allres:
        fig, axes = plt.subplots(len(sel), 4, figsize=(8, 2 * len(sel)))
        for i, r in enumerate(sel):
            img = np.asarray(C.load_rgb(r["path"]))
            pr = allres[k][r["name"]]
            overlay(axes[i, 0], img, None, f"{r['name']} y={r['label']} p(PD)={pr['prob_pd']:.2f}")
            for j, m in enumerate(METHODS):
                overlay(axes[i, j + 1], img, cams_orig[(k, r["name"], m)][0], m)
        fig.suptitle(k, fontsize=9); fig.tight_layout()
        fig.savefig(fig_dir / f"{k}_samples.png", dpi=110); plt.close(fig)
    if len(allres) == 2:
        sp = [r for r in rows if r["task"] == "spiral"][::3]
        fig, axes = plt.subplots(len(sp), 3, figsize=(6, 2 * len(sp)))
        for i, r in enumerate(sp):
            img = np.asarray(C.load_rgb(r["path"]))
            overlay(axes[i, 0], img, None, f"{r['name']} y={r['label']}")
            for j, k in enumerate(allres):
                pr = allres[k][r["name"]]
                overlay(axes[i, j + 1], img, cams_orig[(k, r["name"], "gradcam")][0],
                        f"{k.split('_')[0]} Grad-CAM p(PD)={pr['prob_pd']:.2f}")
        fig.tight_layout(); fig.savefig(fig_dir / "A1_vs_A3_gradcam_spiral.png", dpi=110); plt.close(fig)
    np.savez_compressed(OUT / "cams_input_frame.npz",
                        **{f"{k}__{n}__{m}": allres[k][n]["cams"][m] for k in allres for n in names for m in METHODS})
    print("done ->", OUT)


if __name__ == "__main__":
    main()
