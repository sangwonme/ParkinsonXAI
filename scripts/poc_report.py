"""Collect PoC results into results/poc/poc_results.md + figures for the paper.
Run: MPLBACKEND=Agg .venv/Scripts/python.exe scripts/poc_report.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"; os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("XAIPD_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xaipd import memguard as MG  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from xaipd import symptoms as SY  # noqa: E402
from xaipd.common import RESULTS, ROOT  # noqa: E402

P = RESULTS / "poc"; FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
S = json.load(open(P / "symptom_summary.json"))
B = json.load(open(P / "blackbox_summary.json")) if (P / "blackbox_summary.json").exists() else None
ABL = {t: json.load(open(P / f"symptom_summary_{t}.json")) for t in ("fixedops", "nores", "lam002") if (P / f"symptom_summary_{t}.json").exists()}
sabo = json.load(open(RESULTS / "gait_motion" / "sabo_summary.json"))
sabo_f = json.load(open(RESULTS / "gait_motion" / "sabo_followup.json"))


def fmt(m, ci=None, k="macro_f1"):
    s = f"{m[k]:.3f}"
    return s + (f" [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if ci and k in ci else "")


rows = []
for name, d in (("Proposed symptom model (ours)", S["proposed"]), ("Logistic regression on gait features", S["logreg_features"]),
                ("Random forest on gait features", S["random_forest_features"])):
    rows.append((name, fmt(d["metrics"], d["ci95"], "acc"), fmt(d["metrics"], d["ci95"]), fmt(d["metrics"], d["ci95"], "qwk"), f"{d['metrics']['mae_expected']:.3f}"))
if B:
    rows.append(("Black-box joint x time CNN", fmt(B["metrics"], B["ci95"], "acc"), fmt(B["metrics"], B["ci95"]), fmt(B["metrics"], B["ci95"], "qwk"), f"{B['metrics']['mae_expected']:.3f}"))
for t, d in ABL.items():
    lab = {"fixedops": "Ablation: operators frozen", "nores": "Ablation: no residual", "lam002": "Ablation: weak residual penalty (0.02)"}[t]
    rows.append((lab, fmt(d["proposed"]["metrics"], d["proposed"]["ci95"], "acc"), fmt(d["proposed"]["metrics"], d["proposed"]["ci95"]),
                 fmt(d["proposed"]["metrics"], d["proposed"]["ci95"], "qwk"), f"{d['proposed']['metrics']['mae_expected']:.3f}"))

md = ["# PoC results (CARE-PD, 2953 walks, 5-fold subject-grouped CV, classes 0 / 1 / 2+)", "",
      "## Accuracy (95% bootstrap CI over walks)", "", "| model | accuracy | macro-F1 | quadratic kappa | MAE (expected score) |", "|---|---|---|---|---|"]
md += [f"| {' | '.join(r)} |" for r in rows]
md += ["", f"Residual share |r| / (sum|e_k| + |r|): mean {S['residual_share']['mean']:.3f}, median {S['residual_share']['median']:.3f}, 90th pct {S['residual_share']['p90']:.3f}"]
for t, d in ABL.items():
    md.append(f"- {t}: residual share mean {d['residual_share']['mean']:.3f}")
md += ["", "## Symptom injection into healthy test walks (n per row = all class-0 test walks)", "",
       "| model + explanation | injection | mean change in expected score | share of walks whose class went up | explanation points at the injected symptom | chance |", "|---|---|---|---|---|---|"]
INJ = [("proposed", "ours: contributions e_k"), ("logreg", "logistic regression: coefficient x change"), ("random_forest", "random forest: TreeSHAP")]
for mkey, lab in INJ:
    for kind in ("reduced_arm_swing", "short_stride", "arm_asymmetry"):
        v = S["injection"][f"{mkey}|{kind}"]
        md.append(f"| {lab} | {kind} | {v['mean_d_expected_severity']:.3f} | {v['class_increase_rate']:.3f} | {v['explanation_hit_rate']:.3f} | {v['chance_hit']:.3f} |")
if B:
    for mt in ("gradcam", "ig"):
        for kind in ("reduced_arm_swing", "short_stride", "arm_asymmetry"):
            v = B["injection"][f"blackbox_{mt}|{kind}"]
            md.append(f"| black-box CNN: {'Grad-CAM' if mt == 'gradcam' else 'Integrated Gradients'} (joint group) | {kind} | {v['mean_d_expected_severity']:.3f} | {v['class_increase_rate']:.3f} | {v['explanation_hit_rate']:.3f} | {v['chance_hit']:.3f} |")
md += ["", "## Robustness: share of walks whose predicted class changes", "", "| model | 60 zero frames prepended | body scale x0.8 | body scale x1.25 |", "|---|---|---|---|"]
for m in ("proposed", "logreg"):
    md.append(f"| {m} | {S['robustness'][f'{m}|pad60']:.3f} | {S['robustness'][f'{m}|scale0.8']:.3f} | {S['robustness'][f'{m}|scale1.25']:.3f} |")
if B:
    md.append(f"| black-box CNN | {B['robustness']['blackbox|pad60']:.3f} | {B['robustness']['blackbox|scale0.8']:.3f} | {B['robustness']['blackbox|scale1.25']:.3f} |")
pc = sabo_f["padding_carepd"]; sc = sabo_f["scale_sensitivity_frontal"]
md += [f"| Sabo ST-GCN public weights (reference; 2D projection; padding: zero vs none, 60-frame walks) | {pc['k60|none']['class_change_rate_vs_zero_begin']:.3f} | torso 190->150: Spearman {sc['190']['spearman']:.2f} -> {sc['150']['spearman']:.2f} | torso 190->230: {sc['190']['spearman']:.2f} -> {sc['230']['spearman']:.2f} |"]
md += ["", "## Learned operator parameters (fold 0)", "", "```", json.dumps(S["learned_operator_params"][0]), "```",
       "", "Mean |contribution| per symptom (test walks): " + json.dumps(S["mean_abs_contribution"]), "",
       f"Runtime proposed+classic: {S['runtime_s']} s CPU, peak RSS {S['peak_rss_gb']} GB." + (f" Black-box: {B['runtime_s']} s on {B['device']}." if B else "")]
(P / "poc_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

# ---------------- figure 1: learned shape functions (fold 0)
m = SY.SymptomModel(); m.load_state_dict(load_file(str(ROOT / "models" / "safe" / "poc_symptom" / "fold0.safetensors"))); m.eval()
d = np.load(P / "symptom_test_outputs.npz"); C0 = d["C0"]
fig, axs = plt.subplots(2, 4, figsize=(11, 4.6))
for k, ax in enumerate(axs.ravel()):
    zs = np.linspace(-2.5, 2.5, 200)
    with torch.no_grad():
        g = m.g[k](torch.tensor(zs, dtype=torch.float32)).numpy() - m.g[k](m.ref[k:k + 1]).numpy()
    xs = zs * float(m.sd[k]) + float(m.mu[k])
    ax.plot(xs, g, c="tab:red"); ax.axhline(0, c="k", lw=0.6)
    ax.axvline(float(m.ref[k]) * float(m.sd[k]) + float(m.mu[k]), c="tab:green", ls="--", lw=0.8)
    ax.set_title(SY.NICE[SY.SYMPTOMS[k]], fontsize=9); ax.tick_params(labelsize=7)
    ax.hist(C0[:, k], bins=40, density=True, alpha=0.15, color="0.3", weights=None)
fig.suptitle("Learned monotone shape functions g_k (fold 0): contribution to the severity score vs symptom value\n(dashed: healthy reference; grey: data distribution, rescaled)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "shape_functions.pdf"); fig.savefig(P / "shape_functions.png", dpi=110); plt.close(fig)

# ---------------- figure 2: injection faithfulness
labels, hits, chance = [], [], []
for kind, short in (("reduced_arm_swing", "arm swing"), ("short_stride", "stride"), ("arm_asymmetry", "asymmetry")):
    for mkey, lab in (("proposed", "ours"), ("logreg", "LogReg"), ("random_forest", "RF+SHAP")):
        labels.append(f"{short}\n{lab}"); hits.append(S["injection"][f"{mkey}|{kind}"]["explanation_hit_rate"]); chance.append(S["injection"][f"{mkey}|{kind}"]["chance_hit"])
    if B:
        for mt, lab in (("gradcam", "CNN Grad-CAM"), ("ig", "CNN IG")):
            v = B["injection"][f"blackbox_{mt}|{kind}"]; labels.append(f"{short}\n{lab}"); hits.append(v["explanation_hit_rate"]); chance.append(v["chance_hit"])
fig, ax = plt.subplots(figsize=(11, 3.2))
cols = ["tab:red" if "ours" in l else ("tab:gray" if "CNN" in l else "tab:blue") for l in labels]
ax.bar(range(len(hits)), hits, color=cols); ax.scatter(range(len(hits)), chance, marker="_", s=300, c="k", label="chance")
ax.set_xticks(range(len(hits))); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylabel("explanation points at\ninjected symptom", fontsize=8); ax.set_ylim(0, 1.05); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "injection.pdf"); fig.savefig(P / "injection.png", dpi=110); plt.close(fig)
# copy earlier motivation figures for the paper
import shutil  # noqa: E402
for src, dst in ((RESULTS / "gait_motion" / "animations" / "skeleton_montage.png", "sabo_montage.png"),
                 (RESULTS / "gait_motion" / "figures" / "sabo_scale_sensitivity.png", "sabo_scale.png")):
    if src.exists():
        shutil.copy(src, FIG / dst)
MG.report(" poc_report")

# ---------------- LaTeX numbers and tables for paper/main.tex
PAPER = ROOT / "paper"
sp = sabo["carepd_frontal"]; go = sabo_f["group_occlusion"]["carepd_frontal"]
num = {
    "PropFone": f"{S['proposed']['metrics']['macro_f1']:.2f}", "LRFone": f"{S['logreg_features']['metrics']['macro_f1']:.2f}",
    "RFFone": f"{S['random_forest_features']['metrics']['macro_f1']:.2f}", "BBFone": f"{B['metrics']['macro_f1']:.2f}" if B else "n/a",
    "Rho": f"{100 * S['residual_share']['mean']:.0f}\%",
    "SaboArmShare": f"{100 * sp['gradcam']['share_arms(el,wr)']:.0f}\%",
    "SaboLegChange": f"{100 * go['legs']['class_change_rate']:.0f}\%", "SaboArmChange": f"{100 * go['arms']['class_change_rate']:.0f}\%",
    "SaboGcOcc": f"{sp['method_agreement']['gradcam-vs-occlusion']:.2f}",
    "SaboPadChange": f"{100 * sabo_f['padding_carepd']['k60|none']['class_change_rate_vs_zero_begin']:.0f}\%",
    "SaboScaleLo": f"{sabo_f['scale_sensitivity_frontal']['100']['spearman']:.2f}", "SaboScaleHi": f"{sabo_f['scale_sensitivity_frontal']['260']['spearman']:.2f}",
    "BBPad": f"{100 * B['robustness']['blackbox|pad60']:.0f}\%" if B else "n/a",
    "BBScale": f"{100 * B['robustness']['blackbox|scale1.25']:.0f}\%" if B else "n/a",
}
(PAPER / "numbers.tex").write_text("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in num.items()) + "\n", encoding="utf-8")


def ci(d, k):
    return f"{d['metrics'][k]:.2f} ({d['ci95'][k][0]:.2f}--{d['ci95'][k][1]:.2f})"


acc_rows = [("Ours: symptom model", S["proposed"]), ("Logistic regression, same features", S["logreg_features"]),
            ("Random forest, same features", S["random_forest_features"])]
BB_EXTRA = {t: json.load(open(P / f"blackbox_summary{t}.json")) for t in ("_cnn_s1", "_cnn_s2", "_stgcn") if (P / f"blackbox_summary{t}.json").exists()}
if B:
    acc_rows.append(("Black-box CNN (seed 0)", B))
for t, lab in (("_cnn_s1", "Black-box CNN (seed 1)"), ("_cnn_s2", "Black-box CNN (seed 2)"), ("_stgcn", "Black-box graph network (ST-GCN)")):
    if t in BB_EXTRA:
        acc_rows.append((lab, BB_EXTRA[t]))
for t, d in ABL.items():
    acc_rows.append(({"fixedops": "Ablation: operators fixed", "nores": "Ablation: no residual", "lam002": "Ablation: weak residual penalty"}[t], d["proposed"]))
tab = [r"\begin{table}[t]\centering\small\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lccc@{}}\toprule", r"model & macro-F1 & accuracy & weighted kappa \\ \midrule"]
tab += [f"{n} & {ci(d, 'macro_f1')} & {ci(d, 'acc')} & {ci(d, 'qwk')} \\\\" for n, d in acc_rows]
tab += [r"\bottomrule\end{tabular}}", "\caption{Five-fold, person-grouped results on 2953 CARE-PD walks (grades 0, 1, 2+). Brackets: 95\% bootstrap interval.}",
        "\label{tab:acc}\end{table}"]
(PAPER / "tab_accuracy.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")

inj_rows = [("Ours ($e_k$)", "proposed", S), ("Logistic regression", "logreg", S), ("Random forest + SHAP", "random_forest", S)]
tab = [r"\begin{table}[t]\centering\small\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}lccc@{}}\toprule",
       r"explanation & arm swing & stride & asymmetry \\ \midrule"]
for lab, key, src in inj_rows:
    v = [src["injection"][f"{key}|{k}"] for k in ("reduced_arm_swing", "short_stride", "arm_asymmetry")]
    tab.append(f"{lab} & " + " & ".join(f"{x['explanation_hit_rate']:.2f} / {x['class_increase_rate']:.2f}" for x in v) + " \\\\")
for src_, name_ in ((B, "CNN"), (BB_EXTRA.get("_stgcn"), "ST-GCN")):
    if not src_:
        continue
    for mt, lab in (("gradcam", f"{name_} + Grad-CAM"), ("ig", f"{name_} + Integr. Gradients")):
        v = [src_["injection"][f"blackbox_{mt}|{k}"] for k in ("reduced_arm_swing", "short_stride", "arm_asymmetry")]
        tab.append(f"{lab} & " + " & ".join(f"{x['explanation_hit_rate']:.2f} / {x['class_increase_rate']:.2f}" for x in v) + " \\\\")
tab += [r"\bottomrule\end{tabular}}",
        "\caption{Symptom injection into all healthy test walks. First number: share of walks where the explanation points at the injected symptom "
        "(chance: 0.13 for symptoms, 0.25 or 0.20 for joint groups). Second number: share of walks whose predicted grade went up.}",
        "\label{tab:inj}\end{table}"]
(PAPER / "tab_injection.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
print("paper tables written")
extra = {"RhoWeak": f"{100 * ABL['lam002']['residual_share']['mean']:.0f}\%" if "lam002" in ABL else "n/a",
         "GCArm": f"{B['injection']['blackbox_gradcam|reduced_arm_swing']['explanation_hit_rate']:.2f}" if B else "n/a",
         "IGStride": f"{100 * B['injection']['blackbox_ig|short_stride']['explanation_hit_rate']:.0f}\%" if B else "n/a",
         "BBArmUp": f"{100 * B['injection']['blackbox_gradcam|reduced_arm_swing']['class_increase_rate']:.0f}\%" if B else "n/a"}
with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write("\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in extra.items()) + "\n")

extra2 = {"RhoNinety": f"{100 * S['residual_share']['p90']:.0f}" + chr(92) + "%",
          "BBPadRaw": f"{B['robustness']['blackbox|pad60']:.2f}" if B else "n/a",
          "BBScaleLowRaw": f"{B['robustness']['blackbox|scale0.8']:.2f}" if B else "n/a",
          "BBScaleRaw": f"{B['robustness']['blackbox|scale1.25']:.2f}" if B else "n/a"}
with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write("".join(chr(92) + "newcommand{" + chr(92) + k + "}{" + v + "}" + chr(10) for k, v in extra2.items()))


# ---------------- GPU-window additions (if present): more CNN seeds, ST-GCN black box, seed agreement
extra_md = []
for tag, lab in (("_cnn_s1", "Black-box CNN, seed 1"), ("_cnn_s2", "Black-box CNN, seed 2"), ("_stgcn", "Black-box ST-GCN")):
    f = P / f"blackbox_summary{tag}.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    m_, c_ = d["metrics"], d["ci95"]
    extra_md.append(f"| {lab} ({d['device']}) | macro-F1 {m_['macro_f1']:.3f} [{c_['macro_f1'][0]:.3f}, {c_['macro_f1'][1]:.3f}] | kappa {m_['qwk']:.3f} | "
                    f"pad {d['robustness']['blackbox|pad60']:.2f} / x0.8 {d['robustness']['blackbox|scale0.8']:.2f} / x1.25 {d['robustness']['blackbox|scale1.25']:.2f} | "
                    + " ; ".join(f"{k.split('|')[0].split('_')[-1]} {k.split('|')[1]} hit {v['explanation_hit_rate']:.2f}" for k, v in d["injection"].items()) + " |")
from scipy.stats import spearmanr  # noqa: E402
agree = {}
for mt in ("gradcam", "ig"):
    vals = []
    for f in range(5):
        a, b = P / f"bb_attr_healthy_fold{f}_cnn_s1.npz", P / f"bb_attr_healthy_fold{f}_cnn_s2.npz"
        if a.exists() and b.exists():
            A, Bq = np.load(a)[mt], np.load(b)[mt]
            vals += [spearmanr(A[i], Bq[i]).correlation for i in range(len(A))]
    if vals:
        agree[mt] = float(np.nanmean(vals))
if extra_md or agree:
    txt = ["", "## GPU-window additions", "", "| model | accuracy | kappa | robustness (grade change) | injection hit rates |", "|---|---|---|---|---|"] + extra_md
    if agree:
        txt += ["", "Seed agreement of black-box CNN joint attributions on healthy test walks (seed 1 vs seed 2, mean Spearman over 13 joints): "
                + ", ".join(f"{k} {v:.2f}" for k, v in agree.items())]
    with open(P / "poc_results.md", "a", encoding="utf-8") as fh:
        fh.write(chr(10).join(txt) + chr(10))
    print(chr(10).join(txt))

st = BB_EXTRA.get("_stgcn")
cnn_f1 = [d["metrics"]["macro_f1"] for d in [B] + [BB_EXTRA[t] for t in ("_cnn_s1", "_cnn_s2") if t in BB_EXTRA] if d]
extra3 = {
    "CNNFoneLo": f"{min(cnn_f1):.2f}", "CNNFoneHi": f"{max(cnn_f1):.2f}",
    "STFone": f"{st['metrics']['macro_f1']:.2f}" if st else "n/a",
    "STPad": f"{100 * st['robustness']['blackbox|pad60']:.0f}" + chr(92) + "%" if st else "n/a",
    "STScale": f"{100 * st['robustness']['blackbox|scale1.25']:.0f}" + chr(92) + "%" if st else "n/a",
    "STGcArm": f"{st['injection']['blackbox_gradcam|reduced_arm_swing']['explanation_hit_rate']:.2f}" if st else "n/a",
    "STGcStride": f"{st['injection']['blackbox_gradcam|short_stride']['explanation_hit_rate']:.2f}" if st else "n/a",
    "STGcAsym": f"{st['injection']['blackbox_gradcam|arm_asymmetry']['explanation_hit_rate']:.2f}" if st else "n/a",
    "STIgArm": f"{st['injection']['blackbox_ig|reduced_arm_swing']['explanation_hit_rate']:.2f}" if st else "n/a",
    "STPadRaw": f"{st['robustness']['blackbox|pad60']:.2f}" if st else "n/a",
    "STScaleLowRaw": f"{st['robustness']['blackbox|scale0.8']:.2f}" if st else "n/a",
    "STScaleRaw": f"{st['robustness']['blackbox|scale1.25']:.2f}" if st else "n/a",
    "SeedGC": f"{agree.get('gradcam', float('nan')):.2f}", "SeedIG": f"{agree.get('ig', float('nan')):.2f}",
}
with open(PAPER / "numbers.tex", "a", encoding="utf-8") as fh:
    fh.write("".join(chr(92) + "newcommand{" + chr(92) + k + "}{" + v + "}" + chr(10) for k, v in extra3.items()))
