"""Reproduction training (NOT public weights) of El Maachi et al. 2020 parallel 1D-CNN on gaitpdb.
Subject-level split, 3 seeds, CPU only (keeps the shared GPU free). Short schedule.
Run: .venv/Scripts/python.exe scripts/gait_train.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402  (before numpy/torch)
import numpy as np
import torch
from safetensors.torch import save_file

from xaipd import gait as G
from xaipd.common import RESULTS, ROOT

OUT = RESULTS / "gait"; OUT.mkdir(parents=True, exist_ok=True)
MOUT = ROOT / "models" / "safe" / "g2_elmaachi_repro"; MOUT.mkdir(parents=True, exist_ok=True)
EPOCHS, BS, LR = int(sys.argv[1]) if len(sys.argv) > 1 else 20, 256, 1e-3
MG.limit_torch()

t0 = time.time()
d = G.build_dataset(ROOT / "data" / "processed" / "gaitpdb_seg.npz")
print("segments", d["X"].shape, "subjects", len(np.unique(d["subj"])), f"{time.time()-t0:.0f}s")
tr, te = G.subject_split(d["subj"], d["y"], 0.25, seed=0)
# inner validation split (subjects) from train for epoch selection
tr_idx = np.where(tr)[0]
itr, iva = G.subject_split(d["subj"][tr_idx], d["y"][tr_idx], 0.15, seed=1)
TR, VA, TE = tr_idx[itr], tr_idx[iva], np.where(te)[0]
assert not (set(d["subj"][TR]) & set(d["subj"][TE])) and not (set(d["subj"][VA]) & set(d["subj"][TE]))
split = dict(train=sorted(set(d["subj"][TR])), val=sorted(set(d["subj"][VA])), test=sorted(set(d["subj"][TE])))
json.dump(split, open(OUT / "subject_split.json", "w"), indent=1)
print({k: len(v) for k, v in split.items()})

X = torch.from_numpy(d["X"]); Y = torch.from_numpy(d["y"]).float()
pos_w = torch.tensor((Y[TR] == 0).sum() / (Y[TR] == 1).sum())


def evaluate(model, idx):
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(model(X[idx[i:i + 2048]])) for i in range(0, len(idx), 2048)]).numpy()
    y = d["y"][idx]
    seg_acc = float(((p >= 0.5) == y).mean())
    subj = d["subj"][idx]
    us = np.unique(subj)
    sp = np.array([p[subj == s].mean() for s in us]); sy = np.array([y[subj == s][0] for s in us])
    from sklearn.metrics import roc_auc_score
    return dict(seg_acc=seg_acc, subj_acc=float(((sp >= 0.5) == sy).mean()),
                subj_auc=float(roc_auc_score(sy, sp)), n_subj=int(len(us)),
                subj_sens=float(((sp >= 0.5) & (sy == 1)).sum() / (sy == 1).sum()),
                subj_spec=float(((sp < 0.5) & (sy == 0)).sum() / (sy == 0).sum())), p


summary = {}
for seed in (0, 1, 2):
    torch.manual_seed(seed); np.random.seed(seed)
    model = G.ParallelCNN()
    opt = torch.optim.NAdam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_sd, ts = -1, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = TR[np.random.permutation(len(TR))]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            opt.zero_grad(); loss = lossf(model(X[b]), Y[b]); loss.backward(); opt.step()
        va, _ = evaluate(model, VA)
        if va["seg_acc"] > best:
            best, best_sd = va["seg_acc"], {k: v.clone() for k, v in model.state_dict().items()}
        print(f"seed{seed} ep{ep} loss={loss.item():.3f} val_seg={va['seg_acc']:.3f} val_subj={va['subj_acc']:.3f} {time.time()-ts:.0f}s", flush=True)
    model.load_state_dict(best_sd)
    res, _ = evaluate(model, TE)
    res["train_seconds"] = round(time.time() - ts, 1)
    summary[f"seed{seed}"] = res
    save_file(model.state_dict(), str(MOUT / f"seed{seed}.safetensors"))
    print("TEST", seed, res, flush=True)
json.dump(summary, open(OUT / "g2_e0_accuracy.json", "w"), indent=1)
print("total", f"{time.time()-t0:.0f}s")

MG.report(" gait_train")
