# RESEARCH_CONTEXT — everything a new agent needs (last update 2026-11-08)

## 1. Goal
1. Show with experiments that accurate deep Parkinson's disease (PD) detectors are **not explainable** by the usual
   post-hoc heat maps (Grad-CAM and relatives): the maps point at irrelevant regions, fail sanity checks, are
   unfaithful in deletion tests, and hide shortcuts (padding, scale).
2. Propose and test a gait model that is **explainable by design**: prediction = sum of contributions of measured
   symptoms + a small penalised residual (PoC on CARE-PD).

## 2. Timeline (2026-11-07 .. 11-08)
| step | what | outcome |
|---|---|---|
| plan | survey of CNN PD detectors per modality (drawings, DaTscan, MRI, voice) and public weights | DaTscan/MRI need PPMI DUA or weights are undocumented -> start with drawings |
| user change 1 | "no training, use public pretrained weights" | drawings: Parkonix MobileNet (Keras h5) + Swin-T (safetensors); fastai pickle never opened |
| M1 drawings | E0 accuracy, Grad-CAM/++/Score-CAM, stroke attention, consistency, sanity check, deletion | weak stroke focus, models/methods disagree, partial sanity-check failure (details §5.1) |
| user change 2 | add gait modality | first PhysioNet gaitpdb (foot pressure) -> **dropped**: user wants movement (pose), not force plates (data kept on disk) |
| gait motion | public pose models with weights: only Sabo et al. ST-GCN (10 folds, Detectron 2D, UPDRS-gait) | tested on its 8 usable public samples + CARE-PD external (SMPL -> 2D) |
| findings | Grad-CAM vs occlusion disagree, padding shortcut, scale sensitivity | §5.2; skeleton animations made for presentation |
| PoC (this phase) | symptom-operator model vs black-box CNN vs classic ML on CARE-PD | §5.3 |

## 3. Abandoned directions and why
- DaTscan SPECT (DrivenData DaT challenge weights): needs competition data or PPMI DUA -> not pursued.
- MRI Hugging Face Keras models: no model cards (data/preprocessing unknown) -> irreproducible.
- Voice: no public CNN weights found.
- A2 fastai ResNet-34 (`.pkl`): pickle; isolated environment not approved -> never unpickled.
- gaitpdb (foot pressure, 1D-CNN reproduction): dropped by user in favour of pose/movement data.
- CARE-PD CNN reproduction (old `scripts/carepd_cnn.py`): killed by a system low-memory cleanup while waiting for the GPU;
  superseded by `scripts/poc_blackbox.py`.
- MotionBERT frozen encoder + linear probe: not done (needs CARE-PD Google-Drive checkpoints + SMPL->H36M conversion).

## 4. Confirmed direction: model
```
s_hat(x) = b + sum_k e_k(x) + r(x),     e_k = g_k(c_k(x; theta_k)) - g_k(c_k_ref)
P(y > j | x) = sigmoid(s_hat(x) - tau_j),   tau_1 < tau_2           (grades 0, 1, 2+)
L = -log P(y | x) * w_y + lambda * r(x)^2                           (lambda = 0.5)
rho = |r| / (sum_k |e_k| + |r|)                                     (residual share, reported)
```
- c_k (`src/xaipd/symptoms.py:SymptomOperators`): stride length, cadence, arm swing, arm-swing asymmetry,
  rhythm irregularity (spectral entropy), trunk flexion, walking speed, freezing index. Learnable: gait band,
  freezing / locomotion bands, wrist-vs-elbow mix, spectral temperature. All frames with all-zero joints are masked
  (padding-invariant); all lengths / leg length (scale-invariant). Verified exactly invariant (0 change).
- g_k (`MonotoneShape`): sum of positive sigmoids x clinical sign (+/-; cadence sign learned) -> monotone by
  construction, so the monotonicity penalty mu is 0 by design.
- r (`Residual`): 2-layer 1D CNN with 8 channels on pelvis-centred, leg-normalised joints.
- Input: CARE-PD SMPL -> COCO-13 3D joints via forward kinematics (betas = 0, nose = vertex 332 on the head joint).

## 5. Results (sources in brackets)
### 5.1 Drawings (public weights; Kaggle spiral/wave test, 60 images) [`results/target1/*.json`, PLAN.md 7b/7c]
- Accuracy: MobileNet spiral 0.80 / wave 0.57 (claims 83%); Swin-T spiral 0.83 (reported 0.91 on 11 images).
- Stroke attention barely above a random map; at 7x7 resolution the oracle ceiling is low (spiral 1.23).
- Grad-CAM A1 vs A3 Spearman 0.33 (spiral) / 0.09 (wave); Swin Grad-CAM vs Score-CAM -0.55.
- Sanity check: fully permuted MobileNet keeps |rho| 0.26 with the original map; Swin fully permuted = undefined (all-zero maps).
- Deletion: CAM better than random on spirals, not on waves (also in log-odds and correct-only analyses).

### 5.2 Gait, Sabo ST-GCN public weights [`results/gait_motion/*.json`, PLAN.md 7d/7e]
- Public samples: 8 usable walks (all 4 detectron labelled 0) -> accuracy uninformative (0.5).
- CARE-PD external (2953 walks, 2D projection): acc 0.41, macro-F1 0.24, Spearman -0.19; Spearman goes +0.49 -> -0.26 as the
  torso normalisation goes 100 -> 260 (input-scale fragility, not "model performance").
- Grad-CAM 92% arms / 3% legs; occluding legs changes 94% of grades vs arms 54%; Grad-CAM vs occlusion Spearman -0.36.
- Padding: zero-padded frames get 34-79% of attribution; removing/changing padding changes 14-18% of grades.
- Animations: `results/gait_motion/animations/` (GIF/MP4 + montage).

### 5.3 PoC on CARE-PD [`results/poc/poc_results.md` (authoritative table), `results/poc/*_summary*.json`]
See `results/poc/poc_results.md` for all numbers with 95% bootstrap intervals; headline numbers are repeated in
`paper/main.pdf` (generated from the same JSON by `scripts/poc_report.py`). Headline (2953 walks, 5-fold person-grouped):

| model | macro-F1 (95% CI) | weighted kappa |
|---|---|---|
| proposed symptom model | 0.61 (0.59-0.63) | 0.62 |
| logistic regression, same features | 0.66 (0.64-0.68) | 0.65 |
| random forest, same features | 0.64 (0.62-0.66) | 0.62 |
| black-box joint x time CNN (seed 0, CPU) | 0.58 (0.56-0.60) | 0.52 |
| ablations: frozen operators / no residual / weak residual penalty | 0.61 / 0.61 / 0.61 | ~0.62 |

- Residual share: mean 2%, 90th pct 4% (weak penalty 0.02 -> 6%, no accuracy gain).
- Learnable operator parameters gave **no accuracy gain** (frozen 0.606 vs learned 0.610).
- Injection (explanation points at injected symptom / grade went up): ours arm 1.00/0.13, stride 1.00/0.96,
  asymmetry 0.58/0.20; logistic 1.00, 1.00, 0.23; RF+SHAP 0.94, 1.00, 0.38; CNN Grad-CAM 0.29, 0.60, 0.17
  (chance 0.25/0.25/0.20); CNN IG 0.19, 0.03, 0.20. Hit rates of feature-based models are partly built in.
- Robustness (grade change): ours/logistic 0 for padding and x0.8/x1.25 scale (by construction); CNN 0.23 / 0.34 / 0.39.
- Demo video `results/poc/demo/poc_demo.mp4` (36 s, 0.5 MB), paper `paper/main.pdf` (4 pages).
- GPU window (03:00-03:04, RTX 4070): CNN seeds 1-2 macro-F1 0.54 / 0.54 (seed 0 on CPU 0.58 -> seed range
  0.54-0.58); ST-GCN black box 0.60 (0.58-0.62), robustness pad 0.34 / x0.8 0.48 / x1.25 0.80.
  ST-GCN Grad-CAM hits 0.61 / 0.76 / 0.68 (better than CNN) but for asymmetry the grade rises in only 6% while the
  map moves in 68% -> maps track input change, not decision change. CNN seed agreement of joint attributions
  (Spearman): Grad-CAM 0.41, IG 0.75. Sources: `results/poc/blackbox_summary_{_cnn_s1,_cnn_s2,_stgcn}.json`.

## 6. Limitations (keep these in every write-up)
- Joints come from SMPL fits, not from a video pose detector; 2D projections for the Sabo model are orthographic.
- Symptom injection is synthetic; our operators measure exactly what is injected, so high hit rates are partly built in.
- One dataset (CARE-PD), grades merged to 0/1/2+, no leave-one-cohort-out test yet.
- Sabo public samples are few and likely in its training folds.
- Proposed model is not more accurate than logistic regression on the same features.

## 7. Next steps
1. Leave-one-cohort-out evaluation; tune learnable operators / ordinal head.
2. Video pose input (detector noise) and a turning / freezing operator.
3. Residual leakage analysis (does r re-encode symptom information?).
4. MotionBERT linear-probe baseline; reader study with clinicians (hypothesis: contribution bars beat heat maps).

## 8. File map
| path | what |
|---|---|
| `src/xaipd/common.py` | drawing helpers, metrics, CSV/bootstrap utils |
| `src/xaipd/memguard.py` | 6 GB RSS guard, thread limits, BelowNormal priority (import first) |
| `src/xaipd/safe_pickle.py` | static pickle scan + allow-list unpickler (CARE-PD, SMPL with chumpy stubs) |
| `src/xaipd/a1_keras.py`, `a3_swin.py` | drawing models (Grad-CAM/++/Score-CAM) |
| `src/xaipd/sabo.py` | Sabo ST-GCN port (strict key match, weights_only) + CSV pipeline |
| `src/xaipd/care_pd.py` | SMPL FK -> COCO-13, 2D views, Sabo normalisation |
| `src/xaipd/symptoms.py` | **PoC model**: operators, monotone shapes, residual, injections, data cache |
| `src/xaipd/gait.py` | gaitpdb pipeline (dropped direction) |
| `scripts/run_target1.py`, `images_e5_e6.py`, `e6_a1_wave_followup.py` | drawing experiments |
| `scripts/sabo_xai.py`, `sabo_followup.py`, `sabo_skeleton_anim.py` | Sabo gait experiments + animations |
| `scripts/poc_symptom.py` | PoC: proposed + logistic regression + random forest, injection, robustness, ablations (env flags) |
| `scripts/poc_blackbox.py` | PoC: black-box CNN + Grad-CAM / IG on the same folds |
| `scripts/poc_report.py` | tables (`results/poc/poc_results.md`), paper numbers/tables/figures |
| `scripts/poc_demo_video.py` | side-by-side demo video |
| `scripts/repo_audit.py` | pre-publication size/secret audit -> `results/repo_audit.md` |
| `paper/` | LaTeX draft (`main.tex`, generated `numbers.tex`, `tab_*.tex`, `figures/`) |
| `docs/literature_check.md` | related-work check with verified links |
| `PLAN.md`, `NOTES.md` | Korean lab notebook (full history, run times, licences, incidents) |

## 9. Operating rules
See `AGENTS.md` (licences, untrusted downloads, weights_only, memory guard, .gpu_busy, headless, no commit/push without approval).
