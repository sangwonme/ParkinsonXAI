"""Drawings: E5 cascading model-randomisation sanity check, E6 deletion/insertion, and Swin Score-CAM
target-layer recheck. CPU only (shared GPU). Grad-CAM is the explanation under test.
Run: .venv/Scripts/python.exe scripts/images_e5_e6.py"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402  (must precede numpy/torch/tf)
import numpy as np
from PIL import Image

from xaipd import common as C

OUT = C.RESULTS / "target1"
RNG = np.random.default_rng(0)
STEPS = np.linspace(0, 1, 11)


def auc(y):
    return float(np.trapz(y, STEPS))


def del_ins(score_fn, x, white, rank):
    """x (H,W,3) model-space input; white = model-space value of blank paper; rank = flat pixel order
    (most important first). Returns deletion & insertion curves (prob of fixed class)."""
    H, W = x.shape[:2]
    n = H * W
    dels, inss = [], []
    batch_d, batch_i = [], []
    for s in STEPS:
        k = int(round(s * n))
        m = np.zeros(n, bool); m[rank[:k]] = True; m = m.reshape(H, W)
        xd = x.copy(); xd[m] = white
        xi = np.broadcast_to(white, x.shape).copy(); xi[m] = x[m]
        batch_d.append(xd); batch_i.append(xi)
    return score_fn(np.stack(batch_d)), score_fn(np.stack(batch_i))


def upsample(cam, size=224):
    return np.asarray(Image.fromarray(cam.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BILINEAR))


def summarize(rows, keys):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


# ---------------------------------------------------------------- A1 (Keras)
def run_a1(rows):
    import tensorflow as tf
    MG.limit_tf()
    from xaipd import a1_keras as A1

    model = A1.load()
    cam_model, _, _ = A1.build_cam_model(model)
    xs = np.stack([C.a1_input(C.load_rgb(r["path"])) for r in rows])
    probs = model.predict(xs, verbose=0)
    preds = probs.argmax(1)
    orig = [A1.gradcam(cam_model, xs[i], int(preds[i])) for i in range(len(rows))]

    # E6
    e6 = []
    for i, r in enumerate(rows):
        cls = int(preds[i])
        f = lambda b: model.predict(b, verbose=0)[:, cls]
        rank_c = np.argsort(-upsample(orig[i]).ravel())
        rank_r = RNG.permutation(224 * 224)
        dc, ic = del_ins(f, xs[i], 1.0, rank_c)
        dr, ir = del_ins(f, xs[i], 1.0, rank_r)
        e6.append(dict(name=r["name"], task=r["task"], del_cam=auc(dc), del_rand=auc(dr),
                       ins_cam=auc(ic), ins_rand=auc(ir)))

    # E5 cascading randomisation (top -> bottom), weights permuted within each tensor
    wlayers = [l for l in A1._flatten_layers(model) if l.get_weights()]
    saved = [l.get_weights() for l in wlayers]
    head = [l for l in wlayers if isinstance(l, tf.keras.layers.Dense)]
    bb = [l for l in wlayers if not isinstance(l, tf.keras.layers.Dense)]
    stages = {"head": head, "head+top25%": head + bb[-len(bb) // 4:],
              "head+top50%": head + bb[-len(bb) // 2:], "all": wlayers}
    e5 = {}
    for name, layers in stages.items():
        for l in layers:
            l.set_weights([RNG.permutation(w.ravel()).reshape(w.shape) for w in l.get_weights()])
        rho = [C.spearman(upsample(orig[i]), upsample(A1.gradcam(cam_model, xs[i], int(preds[i]))))
               for i in range(len(rows))]
        e5[name] = float(np.nanmean(np.abs(rho)))
        for l, w in zip(wlayers, saved):
            l.set_weights(w)
    return e5, e6


# ---------------------------------------------------------------- A3 (Swin, torch CPU)
def run_a3(rows):
    import torch
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from xaipd import a3_swin as A3

    MG.limit_torch()
    model = A3.load("cpu")
    xs = torch.stack([A3.preprocess(C.load_rgb(r["path"])) for r in rows])
    with torch.no_grad():
        probs = torch.softmax(model(xs), -1).numpy()
    preds = probs.argmax(1)
    white = ((1.0 - A3.MEAN) / A3.STD).astype(np.float32)

    def cam_with(layer, cls_list, method=GradCAM, x=xs):
        """ONE image at a time. (Previous version passed all 60 images to Score-CAM at once:
        60 x 768 upsampled 224x224 masks x 3 channels -> tens of GB. That caused the 47 GB commit.)"""
        cam = method(model=model, target_layers=[layer], reshape_transform=A3.reshape_transform)
        if method is ScoreCAM:
            cam.batch_size = 16
        out = []
        for i in range(len(x)):
            out.append(cam(input_tensor=x[i:i + 1], targets=[ClassifierOutputTarget(int(cls_list[i]))])[0].astype(np.float32))
        del cam
        return np.stack(out)

    L_final = model.m.swin.layernorm
    L_alt = model.m.swin.encoder.layers[-1].blocks[-1].layernorm_before
    orig = cam_with(L_final, preds)

    # Score-CAM target-layer recheck (E2 enrichment + agreement with Grad-CAM)
    recheck = {}
    for lname, L in (("swin.layernorm", L_final), ("layers[-1].blocks[-1].layernorm_before", L_alt)):
        gc = cam_with(L, preds, GradCAM); sc = cam_with(L, preds, ScoreCAM)
        enr = {"gradcam": [], "scorecam": []}; rho = []
        for i, r in enumerate(rows):
            img = C.load_rgb(r["path"]); w, h = img.size; mask = C.stroke_mask(img)
            for k, cm in (("gradcam", gc[i]), ("scorecam", sc[i])):
                enr[k].append(C.stroke_metrics(C.cam_to_original(cm, (w, h), "resize"), mask)["enrichment"])
            rho.append(C.spearman(gc[i], sc[i]))
        recheck[lname] = dict(gradcam_enrichment=float(np.nanmean(enr["gradcam"])),
                              scorecam_enrichment=float(np.nanmean(enr["scorecam"])),
                              gradcam_vs_scorecam_spearman=float(np.nanmean(rho)))

    # E6
    e6 = []
    xs_np = xs.numpy().transpose(0, 2, 3, 1)
    for i, r in enumerate(rows):
        cls = int(preds[i])

        def f(b):
            with torch.no_grad():
                return torch.softmax(model(torch.from_numpy(b.transpose(0, 3, 1, 2).copy())), -1)[:, cls].numpy()
        dc, ic = del_ins(f, xs_np[i], white, np.argsort(-orig[i].ravel()))
        dr, ir = del_ins(f, xs_np[i], white, RNG.permutation(224 * 224))
        e6.append(dict(name=r["name"], task=r["task"], del_cam=auc(dc), del_rand=auc(dr),
                       ins_cam=auc(ic), ins_rand=auc(ir)))

    # E5 cascading randomisation: classifier -> stage4 -> stage3 -> all
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    sw = model.m.swin
    groups = {"head": [model.m.classifier, sw.layernorm],
              "head+stage4": [model.m.classifier, sw.layernorm, sw.encoder.layers[3]],
              "head+stage3-4": [model.m.classifier, sw.layernorm, sw.encoder.layers[3], sw.encoder.layers[2]],
              "all": [model.m]}
    e5 = {}
    g = torch.Generator().manual_seed(0)
    for name, mods in groups.items():
        with torch.no_grad():
            for m in mods:
                for p in m.parameters():
                    p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g)].view_as(p))
        rnd = cam_with(L_final, preds)
        e5[name] = float(np.nanmean([abs(C.spearman(orig[i], rnd[i])) for i in range(len(rows))]))
        model.load_state_dict(sd)
    return e5, e6, recheck


def main():
    t0 = time.time()
    rows = C.list_split("testing")
    if len(sys.argv) > 1:  # smoke test: N images per task
        n = int(sys.argv[1])
        rows = [r for r in rows if r["task"] == "spiral"][:n] + [r for r in rows if r["task"] == "wave"][:n]
        global OUT
        OUT = C.RESULTS / "target1" / "smoke"; OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    e5a1, e6a1 = run_a1(rows)
    print("A1 done", f"{time.time()-t0:.0f}s", e5a1, flush=True)
    e5a3, e6a3, recheck = run_a3(rows)
    print("A3 done", f"{time.time()-t0:.0f}s", e5a3, recheck, flush=True)
    for k, e5, e6 in (("A1_MobileNet_TM", e5a1, e6a1), ("A3_SwinTiny", e5a3, e6a3)):
        C.write_csv(OUT / f"e6_{k}_per_image.csv", e6)
        res[k] = dict(e5_abs_spearman_vs_orig=e5,
                      e6={t: summarize([r for r in e6 if r["task"] == t], ["del_cam", "del_rand", "ins_cam", "ins_rand"])
                          for t in ("spiral", "wave")})
    res["A3_scorecam_layer_recheck"] = recheck
    res["runtime_s"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
    json.dump(res, open(OUT / "e5_e6.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    MG.report(" images_e5_e6")


if __name__ == "__main__":
    main()
