"""PoC: symptom-operator model (proposed) + gait-feature classic ML baselines on CARE-PD.
5-fold StratifiedGroupKFold by subject (all 4 UPDRS cohorts pooled, classes {0,1,>=2}).
Evaluation: accuracy / macro-F1 / quadratic-weighted kappa / MAE with bootstrap CI; symptom injection
faithfulness; padding & scale robustness; residual share. CPU, BelowNormal priority, 2 threads, 6 GB guard.
Run: XAIPD_THREADS=2 .venv/Scripts/python.exe scripts/poc_symptom.py [epochs]"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("XAIPD_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from xaipd import symptoms as SY  # noqa: E402
from xaipd.common import RESULTS, ROOT, write_csv  # noqa: E402

MG.limit_torch()
OUT = RESULTS / "poc"; OUT.mkdir(parents=True, exist_ok=True)
MOUT = ROOT / "models" / "safe" / "poc_symptom"; MOUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LAMBDA_R = float(os.environ.get("SYM_LAMBDA_R", "0.5"))
FIXED_OPS = os.environ.get("SYM_FIXED_OPS") == "1"   # ablation: operators frozen at their handcrafted init
NO_RES = os.environ.get("SYM_NO_RES") == "1"         # ablation: no residual encoder
TAG = os.environ.get("SYM_TAG", "")
INJ = ["reduced_arm_swing", "short_stride", "arm_asymmetry"]
t0 = time.time()
torch.manual_seed(0); np.random.seed(0)

walks, meta = SY.load_cache()
y = meta["y"]; groups = meta["subj"]; N = len(walks)
lengths = np.array([len(w) for w in walks])


def ops_values(ops, idx_or_walks, bs=96):
    out = []
    with torch.no_grad():
        if isinstance(idx_or_walks, list) and len(idx_or_walks) and isinstance(idx_or_walks[0], np.ndarray):
            W = idx_or_walks; idx = list(range(len(W)))
        else:
            W = walks; idx = list(idx_or_walks)
        for s in range(0, len(idx), bs):
            out.append(ops(SY.batchify(W, idx[s:s + bs])).numpy())
    return np.concatenate(out)


def predict(model, W, idx=None, bs=64):
    idx = list(range(len(W))) if idx is None else list(idx)
    res = {k: [] for k in ("e", "r", "yhat", "p", "c")}
    model.eval()
    with torch.no_grad():
        order = np.argsort(lengths[idx]) if W is walks else np.arange(len(idx))
        tmp = {}
        for s in range(0, len(idx), bs):
            sel = [idx[o] for o in order[s:s + bs]]
            o = model(SY.batchify(W, sel)); p = model.class_probs(o["yhat"])
            for j, i in enumerate(sel):
                tmp[i] = dict(e=o["e"][j].numpy(), r=float(o["r"][j]), yhat=float(o["yhat"][j]), p=p[j].numpy(), c=o["c"][j].numpy())
    for i in idx:
        for k in res:
            res[k].append(tmp[i][k])
    return {k: np.array(v) for k, v in res.items()}


def metrics(yt, P):
    from sklearn.metrics import cohen_kappa_score, f1_score
    yp = P.argmax(1); exp = P @ np.arange(P.shape[1])
    return dict(acc=float((yp == yt).mean()), macro_f1=float(f1_score(yt, yp, average="macro")),
                qwk=float(cohen_kappa_score(yt, yp, weights="quadratic")), mae_expected=float(np.abs(exp - yt).mean()))


def boot_ci(yt, P, n=1000, seed=0):
    rng = np.random.default_rng(seed); vals = {k: [] for k in ("acc", "macro_f1", "qwk", "mae_expected")}
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        for k, v in metrics(yt[i], P[i]).items():
            vals[k].append(v)
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in vals.items()}


# ---------------------------------------------------------------- splits
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = [(tr, te) for tr, te in sgkf.split(np.zeros(N), y, groups)]
json.dump({"folds": [{"test_subjects": sorted(set(groups[te].tolist()))} for _, te in folds]}, open(OUT / "splits.json", "w"), indent=1)
for tr, te in folds:
    assert not set(groups[tr]) & set(groups[te])

ops0 = SY.SymptomOperators()
C0 = ops_values(ops0, range(N))  # fixed (initial) operators = handcrafted gait features
print("features", C0.shape, f"{time.time()-t0:.0f}s", flush=True)

P_sym, P_lr, P_rf = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3))
E_sym, R_sym = np.zeros((N, len(SY.SYMPTOMS))), np.zeros(N)
inj_rows, rob_rows, learned = [], [], []
for f, (tr, te) in enumerate(folds):
    tf = time.time()
    # ---------- classic ML on the same (fixed) features
    mu, sd = C0[tr].mean(0), C0[tr].std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit((C0[tr] - mu) / sd, y[tr])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=f, n_jobs=2).fit(C0[tr], y[tr])
    P_lr[te] = lr.predict_proba((C0[te] - mu) / sd); P_rf[te] = rf.predict_proba(C0[te])

    # ---------- proposed model
    model = SY.SymptomModel(use_residual=not NO_RES)
    if FIXED_OPS:
        for p_ in model.ops.parameters():
            p_.requires_grad_(False)
    model.mu.copy_(torch.tensor(mu, dtype=torch.float32)); model.sd.copy_(torch.tensor(sd, dtype=torch.float32))
    model.ref.copy_(torch.tensor(np.median((C0[tr][y[tr] == 0] - mu) / sd, 0), dtype=torch.float32))
    rng = np.random.default_rng(f)
    tr_subj = np.unique(groups[tr]); val_subj = set(rng.choice(tr_subj, max(1, len(tr_subj) // 10), replace=False))
    va = np.array([i for i in tr if groups[i] in val_subj]); trn = np.array([i for i in tr if groups[i] not in val_subj])
    cw = torch.tensor(len(trn) / (3 * np.bincount(y[trn], minlength=3)), dtype=torch.float32)
    opt = torch.optim.Adam(([] if FIXED_OPS else [{"params": model.ops.parameters(), "lr": 3e-3}]) + [{"params": [p for n, p in model.named_parameters() if not n.startswith("ops.")], "lr": 1e-2}])
    best, best_sd = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(trn)
        perm = perm[np.argsort(lengths[perm] + rng.integers(0, 40, len(perm)))]  # bucket by length (less padding)
        batches = [perm[s:s + 48] for s in range(0, len(perm), 48)]; rng.shuffle(batches)
        for b in batches:
            o = model(SY.batchify(walks, b)); p = model.class_probs(o["yhat"])
            yt = torch.from_numpy(y[b])
            nll = -(torch.log(p.gather(1, yt[:, None])[:, 0]) * cw[yt]).mean()
            loss = nll + LAMBDA_R * (o["r"] ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():  # keep learnable bands ordered and inside [0.1, 12] Hz
                ops = model.ops
                ops.gait_lo.clamp_(0.1, 1.2); ops.gait_hi.clamp_(ops.gait_lo + 0.3, 3.0)
                ops.loco_hi.clamp_(1.0, 4.0); ops.frz_lo.clamp_(2.0, 6.0); ops.frz_hi.clamp_(ops.frz_lo + 1.0, 12.0)
        pv = predict(model, walks, va)
        yv = y[va]; vloss = float(-np.mean(np.log(pv["p"][np.arange(len(va)), yv])))
        if vloss < best:
            best, best_sd = vloss, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"fold{f} ep{ep} val_nll={vloss:.3f} {time.time()-tf:.0f}s", flush=True)
    model.load_state_dict(best_sd)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(MOUT / f"fold{f}{TAG}.safetensors"))
    o = predict(model, walks, te)
    P_sym[te], E_sym[te], R_sym[te] = o["p"], o["e"], o["r"]
    ops = model.ops
    learned.append(dict(fold=f, gait_band_hz=[float(ops.gait_lo), float(ops.gait_hi)], freeze_band_hz=[float(ops.frz_lo), float(ops.frz_hi)],
                        locomotion_band_hz=[0.5, float(ops.loco_hi)], arm_wrist_weight=float(torch.sigmoid(ops.arm_mix)),
                        cadence_sign=float(model.g[1].sign())))

    # ---------- symptom injection on healthy test walks
    healthy = [i for i in te if y[i] == 0]
    W0 = [walks[i] for i in healthy]
    base_sym = predict(model, W0)
    C_base = ops_values(ops0, W0)
    import shap
    expl = shap.TreeExplainer(rf)
    sv0 = expl.shap_values(C_base)
    sv0 = np.stack(sv0, -1) if isinstance(sv0, list) else sv0  # (n, K, classes)
    sev_coef = lr.coef_[2] - lr.coef_[0]
    for kind in INJ:
        Wk = [SY.inject(w, kind) for w in W0]
        inj_sym = predict(model, Wk)
        C_inj = ops_values(ops0, Wk)
        tgt = [SY.SYMPTOMS.index(s) for s in SY.INJ_TARGET[kind]]
        # proposed
        de = inj_sym["e"] - base_sym["e"]
        exp0 = base_sym["p"] @ np.arange(3); exp1 = inj_sym["p"] @ np.arange(3)
        # LR (severity direction contribution), RF (TreeSHAP on expected severity)
        z0, z1 = (C_base - mu) / sd, (C_inj - mu) / sd
        dl = (z1 - z0) * sev_coef[None]
        sv1 = expl.shap_values(C_inj); sv1 = np.stack(sv1, -1) if isinstance(sv1, list) else sv1
        ds_ = (sv1 - sv0) @ np.arange(sv0.shape[-1])
        for name, dexp, d_expl, P0, P1 in (
                ("proposed", exp1 - exp0, de, base_sym["p"], inj_sym["p"]),
                ("logreg", lr.predict_proba(z1) @ np.arange(3) - lr.predict_proba(z0) @ np.arange(3), dl, lr.predict_proba(z0), lr.predict_proba(z1)),
                ("random_forest", rf.predict_proba(C_inj) @ np.arange(3) - rf.predict_proba(C_base) @ np.arange(3), ds_, rf.predict_proba(C_base), rf.predict_proba(C_inj))):
            for j in range(len(healthy)):
                inj_rows.append(dict(fold=f, model=name, injection=kind, walk=str(meta["walk"][healthy[j]]),
                                     d_expected_severity=float(dexp[j]), class_increased=int(P1[j].argmax() > P0[j].argmax()),
                                     top_symptom=SY.SYMPTOMS[int(np.argmax(d_expl[j]))], hit=int(int(np.argmax(d_expl[j])) in tgt)))
    # ---------- robustness: zero padding (60 frames at start) and global scale
    sub = list(te[:200])
    Wsub = [walks[i] for i in sub]
    base = predict(model, Wsub)["p"].argmax(1); bl = lr.predict_proba((ops_values(ops0, Wsub) - mu) / sd).argmax(1)
    for tag, Wt in (("pad60", [np.concatenate([np.zeros((60, 13, 3), np.float32), w]) for w in Wsub]),
                    ("scale0.8", [w * 0.8 for w in Wsub]), ("scale1.25", [w * 1.25 for w in Wsub])):
        rob_rows.append(dict(fold=f, model="proposed", test=tag, class_change_rate=float((predict(model, Wt)["p"].argmax(1) != base).mean())))
        rob_rows.append(dict(fold=f, model="logreg", test=tag, class_change_rate=float((lr.predict_proba((ops_values(ops0, Wt) - mu) / sd).argmax(1) != bl).mean())))
    print(f"fold {f} done {time.time()-tf:.0f}s; total {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- summaries
summary = {"variant": dict(fixed_ops=FIXED_OPS, no_residual=NO_RES, tag=TAG), "n_walks": int(N), "class_counts": np.bincount(y).tolist(), "epochs": EPOCHS, "lambda_r": LAMBDA_R}
for name, P in (("proposed", P_sym), ("logreg_features", P_lr), ("random_forest_features", P_rf)):
    summary[name] = dict(metrics=metrics(y, P), ci95=boot_ci(y, P))
    summary[name]["per_cohort_macro_f1"] = {d: metrics(y[meta["ds"] == d], P[meta["ds"] == d])["macro_f1"] for d in np.unique(meta["ds"])}
rho = np.abs(R_sym) / (np.abs(E_sym).sum(1) + np.abs(R_sym) + 1e-9)
summary["residual_share"] = dict(mean=float(rho.mean()), median=float(np.median(rho)), p90=float(np.quantile(rho, 0.9)))
inj = {}
for name in ("proposed", "logreg", "random_forest"):
    for kind in INJ:
        rr = [r for r in inj_rows if r["model"] == name and r["injection"] == kind]
        inj[f"{name}|{kind}"] = dict(n=len(rr), mean_d_expected_severity=float(np.mean([r["d_expected_severity"] for r in rr])),
                                     class_increase_rate=float(np.mean([r["class_increased"] for r in rr])),
                                     explanation_hit_rate=float(np.mean([r["hit"] for r in rr])),
                                     chance_hit=len(SY.INJ_TARGET[kind]) / len(SY.SYMPTOMS))
summary["injection"] = inj
summary["robustness"] = {f"{m}|{t}": float(np.mean([r["class_change_rate"] for r in rob_rows if r["model"] == m and r["test"] == t]))
                         for m in ("proposed", "logreg") for t in ("pad60", "scale0.8", "scale1.25")}
summary["learned_operator_params"] = learned
summary["mean_abs_contribution"] = dict(zip(SY.SYMPTOMS, np.abs(E_sym).mean(0).round(4).tolist()))
summary["runtime_s"] = round(time.time() - t0, 1); summary["peak_rss_gb"] = round(MG.peak["rss_gb"], 2)
json.dump(summary, open(OUT / f"symptom_summary{TAG}.json", "w"), indent=1)
write_csv(OUT / f"injection_per_walk{TAG}.csv", inj_rows); write_csv(OUT / f"robustness_per_fold{TAG}.csv", rob_rows)
np.savez_compressed(OUT / f"symptom_test_outputs{TAG}.npz", P_sym=P_sym, P_lr=P_lr, P_rf=P_rf, E=E_sym, R=R_sym, y=y, C0=C0)
print(json.dumps({k: v for k, v in summary.items() if k not in ("learned_operator_params",)}, indent=1))
MG.report(" poc_symptom")
