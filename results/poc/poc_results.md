# PoC results (CARE-PD, 2953 walks, 5-fold subject-grouped CV, classes 0 / 1 / 2+)

## Accuracy (95% bootstrap CI over walks)

| model | accuracy | macro-F1 | quadratic kappa | MAE (expected score) |
|---|---|---|---|---|
| Proposed symptom model (ours) | 0.619 [0.602, 0.637] | 0.610 [0.593, 0.629] | 0.616 [0.591, 0.640] | 0.463 |
| Logistic regression on gait features | 0.667 [0.650, 0.684] | 0.661 [0.644, 0.678] | 0.653 [0.629, 0.678] | 0.433 |
| Random forest on gait features | 0.649 [0.633, 0.666] | 0.637 [0.621, 0.655] | 0.621 [0.596, 0.647] | 0.444 |
| Black-box joint x time CNN | 0.597 [0.579, 0.615] | 0.582 [0.563, 0.600] | 0.516 [0.487, 0.544] | 0.484 |
| Ablation: operators frozen | 0.617 [0.598, 0.635] | 0.606 [0.588, 0.623] | 0.618 [0.595, 0.643] | 0.467 |
| Ablation: no residual | 0.618 [0.602, 0.636] | 0.610 [0.592, 0.628] | 0.619 [0.596, 0.643] | 0.462 |
| Ablation: weak residual penalty (0.02) | 0.617 [0.600, 0.636] | 0.606 [0.588, 0.625] | 0.622 [0.599, 0.645] | 0.466 |

Residual share |r| / (sum|e_k| + |r|): mean 0.020, median 0.015, 90th pct 0.041
- fixedops: residual share mean 0.021
- nores: residual share mean 0.000
- lam002: residual share mean 0.061

## Symptom injection into healthy test walks (n per row = all class-0 test walks)

| model + explanation | injection | mean change in expected score | share of walks whose class went up | explanation points at the injected symptom | chance |
|---|---|---|---|---|---|
| ours: contributions e_k | reduced_arm_swing | 0.111 | 0.128 | 1.000 | 0.125 |
| ours: contributions e_k | short_stride | 0.928 | 0.955 | 1.000 | 0.250 |
| ours: contributions e_k | arm_asymmetry | 0.161 | 0.197 | 0.583 | 0.125 |
| logistic regression: coefficient x change | reduced_arm_swing | 0.123 | 0.170 | 1.000 | 0.125 |
| logistic regression: coefficient x change | short_stride | 1.028 | 0.891 | 1.000 | 0.250 |
| logistic regression: coefficient x change | arm_asymmetry | 0.156 | 0.285 | 0.232 | 0.125 |
| random forest: TreeSHAP | reduced_arm_swing | 0.165 | 0.120 | 0.939 | 0.125 |
| random forest: TreeSHAP | short_stride | 0.915 | 0.891 | 1.000 | 0.250 |
| random forest: TreeSHAP | arm_asymmetry | 0.171 | 0.182 | 0.383 | 0.125 |
| black-box CNN: Grad-CAM (joint group) | reduced_arm_swing | 0.044 | 0.114 | 0.288 | 0.250 |
| black-box CNN: Grad-CAM (joint group) | short_stride | 1.023 | 0.747 | 0.601 | 0.250 |
| black-box CNN: Grad-CAM (joint group) | arm_asymmetry | 0.026 | 0.047 | 0.172 | 0.200 |
| black-box CNN: Integrated Gradients (joint group) | reduced_arm_swing | 0.044 | 0.114 | 0.191 | 0.250 |
| black-box CNN: Integrated Gradients (joint group) | short_stride | 1.023 | 0.747 | 0.030 | 0.250 |
| black-box CNN: Integrated Gradients (joint group) | arm_asymmetry | 0.026 | 0.047 | 0.196 | 0.200 |

## Robustness: share of walks whose predicted class changes

| model | 60 zero frames prepended | body scale x0.8 | body scale x1.25 |
|---|---|---|---|
| proposed | 0.000 | 0.000 | 0.000 |
| logreg | 0.000 | 0.000 | 0.000 |
| black-box CNN | 0.225 | 0.338 | 0.391 |
| Sabo ST-GCN public weights (reference; 2D projection; padding: zero vs none, 60-frame walks) | 0.148 | torso 190->150: Spearman -0.19 -> 0.18 | torso 190->230: -0.19 -> -0.25 |

## Learned operator parameters (fold 0)

```
{"fold": 0, "gait_band_hz": [1.0691659450531006, 2.1253466606140137], "freeze_band_hz": [2.906531810760498, 7.975831031799316], "locomotion_band_hz": [0.5, 3.153803825378418], "arm_wrist_weight": 0.8797531127929688, "cadence_sign": -0.9948533177375793}
```

Mean |contribution| per symptom (test walks): {"stride_length": 0.7826, "cadence": 0.2179, "arm_swing": 0.3013, "arm_asymmetry": 0.069, "rhythm_irregularity": 0.0031, "trunk_flexion": 0.2942, "walking_speed": 0.8727, "freeze_index": 0.0038}

Runtime proposed+classic: 278.9 s CPU, peak RSS 1.38 GB. Black-box: 920.9 s on cpu.

## GPU-window additions

| model | accuracy | kappa | robustness (grade change) | injection hit rates |
|---|---|---|---|---|
| Black-box CNN, seed 1 (cuda) | macro-F1 0.541 [0.521, 0.560] | kappa 0.466 | pad 0.23 / x0.8 0.32 / x1.25 0.42 | gradcam reduced_arm_swing hit 0.09 ; gradcam short_stride hit 0.49 ; gradcam arm_asymmetry hit 0.05 ; ig reduced_arm_swing hit 0.11 ; ig short_stride hit 0.02 ; ig arm_asymmetry hit 0.07 |
| Black-box CNN, seed 2 (cuda) | macro-F1 0.543 [0.525, 0.562] | kappa 0.492 | pad 0.26 / x0.8 0.38 / x1.25 0.31 | gradcam reduced_arm_swing hit 0.16 ; gradcam short_stride hit 0.61 ; gradcam arm_asymmetry hit 0.18 ; ig reduced_arm_swing hit 0.20 ; ig short_stride hit 0.15 ; ig arm_asymmetry hit 0.10 |
| Black-box ST-GCN (cuda) | macro-F1 0.601 [0.582, 0.617] | kappa 0.544 | pad 0.34 / x0.8 0.48 / x1.25 0.80 | gradcam reduced_arm_swing hit 0.61 ; gradcam short_stride hit 0.76 ; gradcam arm_asymmetry hit 0.68 ; ig reduced_arm_swing hit 0.22 ; ig short_stride hit 0.56 ; ig arm_asymmetry hit 0.23 |

Seed agreement of black-box CNN joint attributions on healthy test walks (seed 1 vs seed 2, mean Spearman over 13 joints): gradcam 0.41, ig 0.75
