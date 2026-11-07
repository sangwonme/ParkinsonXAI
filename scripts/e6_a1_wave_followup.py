"""Follow-up to E6 for A1 on wave drawings: confidence is saturated (spec 0.13), so probability
deletion curves may be insensitive. Recompute with (a) log-odds of the predicted class and
(b) correctly-classified images only. Grad-CAM order vs 5 random orders. CPU, guarded."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from xaipd import common as C  # noqa: E402

import tensorflow as tf  # noqa: E402

MG.limit_tf()
from xaipd import a1_keras as A1  # noqa: E402

STEPS = np.linspace(0, 1, 11); RNG = np.random.default_rng(0)
model = A1.load(); cam_model, _, _ = A1.build_cam_model(model)
out = {}
for task in ("wave", "spiral"):
    rows = [r for r in C.list_split("testing") if r["task"] == task]
    res = []
    for r in rows:
        x = C.a1_input(C.load_rgb(r["path"]))
        p = model.predict(x[None], verbose=0)[0]; cls = int(p.argmax())
        cam = np.asarray(Image.fromarray(A1.gradcam(cam_model, x, cls).astype(np.float32), mode="F").resize((224, 224), Image.Resampling.BILINEAR))
        orders = {"cam": np.argsort(-cam.ravel())}
        for k in range(5):
            orders[f"rand{k}"] = RNG.permutation(224 * 224)
        curves = {}
        for name, rank in orders.items():
            batch = []
            for s in STEPS:
                m = np.zeros(224 * 224, bool); m[rank[:int(round(s * 224 * 224))]] = True
                xd = x.copy(); xd[m.reshape(224, 224)] = 1.0; batch.append(xd)
            pp = np.clip(model.predict(np.stack(batch), verbose=0)[:, cls], 1e-7, 1 - 1e-7)
            curves[name] = dict(prob=float(np.trapz(pp, STEPS)), logodds=float(np.trapz(np.log(pp / (1 - pp)), STEPS)))
        res.append(dict(correct=int(cls == r["label"]), p_max=float(p.max()),
                        del_prob_cam=curves["cam"]["prob"], del_prob_rand=float(np.mean([curves[f"rand{k}"]["prob"] for k in range(5)])),
                        del_logodds_cam=curves["cam"]["logodds"], del_logodds_rand=float(np.mean([curves[f"rand{k}"]["logodds"] for k in range(5)]))))
    for subset in ("all", "correct_only"):
        rr = [d for d in res if subset == "all" or d["correct"]]
        out[f"{task}|{subset}"] = dict(n=len(rr), mean_pmax=float(np.mean([d["p_max"] for d in rr])),
                                       **{k: float(np.mean([d[k] for d in rr])) for k in ("del_prob_cam", "del_prob_rand", "del_logodds_cam", "del_logodds_rand")})
    print(task, json.dumps({k: v for k, v in out.items() if k.startswith(task)}), flush=True)
out["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(out, open(C.RESULTS / "target1" / "e6_a1_followup.json", "w"), indent=1)
MG.report(" e6_followup")
