"""Cross-modality comparison table + figure: hand drawings (A1 MobileNet, A3 Swin) vs gait motion
(Sabo ST-GCN public weights; CARE-PD joint x time CNN, reproduced training).
Run: .venv/Scripts/python.exe scripts/cross_modality.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xaipd.common import RESULTS, write_csv

T1, GM = RESULTS / "target1", RESULTS / "gait_motion"
OUT = RESULTS / "cross_modality"; OUT.mkdir(parents=True, exist_ok=True)
NA = float("nan")


def jl(p):
    return json.load(open(p)) if Path(p).exists() else None


def rel(x, r, o):
    return (x - r) / (o - r) if (o is not None and r is not None and o != r) else NA


def rnd(v, n=3):
    return round(v, n) if isinstance(v, (int, float)) and not np.isnan(v) else NA


rows = []
e0, e2, orc, e4, e56 = (jl(T1 / f) for f in ("e0_accuracy.json", "e2_stroke_attention.json", "e2_oracle_ceiling.json",
                                             "e4_consistency.json", "e5_e6.json"))
for model, frame, short in (("A1_MobileNet_TM", "A1_frame", "A1 MobileNet (public wts)"), ("A3_SwinTiny", "A3_frame", "A3 Swin-T (public wts, non-CNN)")):
    for task in ("spiral", "wave"):
        g = e2[f"{model}|{task}|gradcam"]; o = orc[f"{frame}|{task}|oracle"]; r = orc[f"{frame}|{task}|random"]
        e6 = e56[model]["e6"][task] if e56 else None
        rows.append(dict(
            modality="drawing", model=short, data=f"Kaggle {task} test (n=30)", accuracy=rnd(e0[model][task]["acc"]),
            roi_def="ink stroke pixels", roi_score_0rand_1oracle=rnd(rel(g["enrichment"], r["enrichment"], o["enrichment"])),
            pointing=rnd(g["pointing_hit"]), pointing_rand=rnd(r["pointing_hit"]), pointing_oracle=rnd(o["pointing_hit"]),
            method_agree_gc_vs_perturb=rnd(e4[f"{model}|gradcam-vs-scorecam|{task}"]["spearman"]),
            model_or_seed_agree=rnd(e4[f"A1-vs-A3|gradcam|{task}"]["spearman"]), agree_kind="A1 vs A3 (diff. models)",
            e5_rho_full_rand=rnd(e56[model]["e5_abs_spearman_vs_orig"]["all"]) if e56 else NA,
            e6_del_diff=rnd(e6["del_cam"] - e6["del_rand"]) if e6 else NA,
            e6_ins_diff=rnd(e6["ins_cam"] - e6["ins_rand"]) if e6 else NA,
            shortcut=f"background energy {g['bg_energy']:.2f}"))

sb = jl(GM / "sabo_summary.json")
if sb:
    for set_name, label in (("public_samples(detectron)", "Sabo public samples (n=4)"), ("carepd_frontal", "CARE-PD frontal (external)")):
        s = sb[set_name]; g = s["gradcam"]; r = s["baseline_random"]; o = s["baseline_oracle"]
        ext = sb["E0_carepd_external_ALL"].get("frontal|190|ALL", {})
        acc = sb["E0_public_samples"]["detectron"]["acc_ensemble"] if "samples" in set_name else ext.get("acc", NA)
        e6 = s["E6"]["gradcam|attr"], s["E6"]["gradcam|rand"]
        rows.append(dict(
            modality="gait motion (2D pose)", model="Sabo ST-GCN (public wts, 10 folds)", data=label, accuracy=rnd(acc),
            roi_def="distal limb joints (stride/arm swing)", roi_score_0rand_1oracle=rnd(rel(g["roi_enrichment"], r["roi_enrichment"], o["roi_enrichment"])),
            pointing=rnd(g["pointing_hit"]), pointing_rand=rnd(r["pointing_hit"]), pointing_oracle=rnd(o["pointing_hit"]),
            method_agree_gc_vs_perturb=rnd(s["method_agreement"]["gradcam-vs-occlusion"]),
            model_or_seed_agree=rnd(s["fold_agreement_gradcam_spearman"]), agree_kind="10 CV folds (same arch)",
            e5_rho_full_rand=rnd(s["E5_abs_spearman_vs_orig"]["all"]["gradcam"]),
            e6_del_diff=rnd(e6[0]["deletion_auc"] - e6[1]["deletion_auc"]), e6_ins_diff=rnd(e6[0]["insertion_auc"] - e6[1]["insertion_auc"]),
            shortcut=f"padding share {g['pad_share']:.2f} (pad frac {g['pad_frac']:.2f}); nose share {g['nose_share']:.2f}; edge enr {g['edge_enrichment']:.2f}"))

cc = jl(GM / "carepd_cnn_e0.json")
if cc and "xai" in cc:
    s = cc["xai"]; g = s["gradcam"]; r = s["baseline_random"]; o = s["baseline_oracle"]; ds = s["dataset"]
    e6 = s["E6"]["gradcam|attr"], s["E6"]["gradcam|rand"]
    rows.append(dict(
        modality="gait motion (3D pose)", model="joint x time CNN (reproduced training, 3 seeds)", data=f"CARE-PD {ds} 6-fold",
        accuracy=rnd(cc[ds]["macro_f1_mean"]), roi_def="distal limb joints",
        roi_score_0rand_1oracle=rnd(rel(g["roi_enrichment"], r["roi_enrichment"], o["roi_enrichment"])),
        pointing=rnd(g["pointing_hit"]), pointing_rand=rnd(r["pointing_hit"]), pointing_oracle=rnd(o["pointing_hit"]),
        method_agree_gc_vs_perturb=rnd(s["method_agreement"]["gradcam-vs-occlusion"]),
        model_or_seed_agree=rnd(s["seed_agreement_gradcam"]), agree_kind="3 seeds (same fold)",
        e5_rho_full_rand=rnd(s["E5_abs_spearman_vs_orig"]["all"]["gradcam"]),
        e6_del_diff=rnd(e6[0]["deletion_auc"] - e6[1]["deletion_auc"]), e6_ins_diff=rnd(e6[0]["insertion_auc"] - e6[1]["insertion_auc"]),
        shortcut=f"nose share {g['nose_share']:.2f}; top joint {g['top_joint_share']:.2f}; edge enr {g['edge_enrichment']:.2f}"))

write_csv(OUT / "cross_modality_table.csv", rows)
cols = list(rows[0].keys())
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r_ in rows:
    md.append("| " + " | ".join("" if (isinstance(r_[c], float) and np.isnan(r_[c])) else str(r_[c]) for c in cols) + " |")
(OUT / "cross_modality_table.md").write_text(
    "# Cross-modality XAI comparison: hand drawing vs gait motion\n\n"
    "Explanation under test = Grad-CAM unless stated. accuracy: drawings = acc; Sabo samples = ensemble acc (UPDRS-gait, all labels 0); "
    "CARE-PD external = acc vs UPDRS clipped to 0-2; CARE-PD CNN = macro-F1 (3 classes).\n\n"
    "- roi_score: 0 = random map, 1 = oracle ROI map (same resolution/geometry)\n"
    "- method_agree: Spearman(Grad-CAM, perturbation method): Score-CAM for images, occlusion for gait\n"
    "- model_or_seed_agree: Spearman between Grad-CAMs of different models (images) or folds/seeds (gait)\n"
    "- e5_rho_full_rand: |Spearman| original vs fully weight-permuted model (high = fails sanity check)\n"
    "- e6 diffs: attribution-order AUC minus random-order AUC (images: prob of predicted class; Sabo: output normalised to [baseline, full]; CNN: prob). deletion < 0 and insertion > 0 = faithful\n\n"
    + "\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

metrics = ["roi_score_0rand_1oracle", "method_agree_gc_vs_perturb", "model_or_seed_agree", "e5_rho_full_rand", "e6_del_diff", "e6_ins_diff"]
labels = [f"{r_['model'].split(' (')[0]} | {r_['data'][:22]}" for r_ in rows]
fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 0.45 * len(rows) + 1.6))
for ax, m in zip(axes, metrics):
    vals = [0 if (isinstance(r_[m], float) and np.isnan(r_[m])) else r_[m] for r_ in rows]
    ax.barh(range(len(rows)), vals, color=["#4C78A8" if r_["modality"] == "drawing" else "#F58518" for r_ in rows])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels if m == metrics[0] else [], fontsize=6)
    for x in {"roi_score_0rand_1oracle": [0, 1], "e6_del_diff": [0], "e6_ins_diff": [0]}.get(m, []):
        ax.axvline(x, ls="--", c="k", lw=0.7)
    ax.set_title(m, fontsize=7); ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=6)
fig.suptitle("Common XAI metrics: drawing (blue) vs gait motion (orange)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "common_metrics_bars.png", dpi=120); plt.close(fig)
print("saved", OUT)
