"""E2 control: what enrichment / pointing would a 'perfect' stroke-focused map achieve at the
same 7x7 CAM resolution? Oracle = stroke mask avg-pooled to 7x7 in each model's input frame,
then mapped back exactly like the real CAMs. Also a uniform-random baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image, ImageOps

from xaipd import common as C


def oracle_cam(img, mode):
    m = Image.fromarray((C.stroke_mask(img) * 255).astype(np.uint8))
    if mode == "fit":
        m = ImageOps.fit(m, (224, 224), Image.Resampling.BILINEAR)
    else:
        m = m.resize((224, 224), Image.Resampling.BILINEAR)
    a = np.asarray(m, np.float32) / 255
    return a.reshape(7, 32, 7, 32).mean(axis=(1, 3))


rng = np.random.default_rng(0)
out = {}
for mode, tag in (("fit", "A1_frame"), ("resize", "A3_frame")):
    for task in ("spiral", "wave"):
        rows = [r for r in C.list_split("testing") if r["task"] == task]
        acc = {"oracle": [], "random": []}
        for r in rows:
            img = C.load_rgb(r["path"])
            w, h = img.size
            mask = C.stroke_mask(img)
            if mode == "fit":
                l, t, rr, b = C.fit_box(w, h)
                valid = np.zeros((h, w), bool); valid[t:b, l:rr] = True
            else:
                valid = np.ones((h, w), bool)
            for k, cam in (("oracle", oracle_cam(img, mode)), ("random", rng.random((7, 7)))):
                acc[k].append(C.stroke_metrics(C.cam_to_original(cam, (w, h), mode), mask, valid))
        for k, lst in acc.items():
            out[f"{tag}|{task}|{k}"] = {s: float(np.nanmean([d[s] for d in lst]))
                                        for s in ("bg_energy", "enrichment", "pointing_hit", "top20_on_stroke")}
json.dump(out, open(C.RESULTS / "target1" / "e2_oracle_ceiling.json", "w"), indent=1)
print(json.dumps(out, indent=1))
