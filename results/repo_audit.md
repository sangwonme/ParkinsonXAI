# Repository audit (pre-publication)

Files that would be committed: 123, total 15.4 MB

## Size by top-level folder

| folder | MB |
|---|---|
| results | 14.37 |
| paper | 0.77 |
| scripts | 0.15 |
| . | 0.08 |
| src | 0.05 |
| docs | 0.00 |
| envs | 0.00 |
| models | 0.00 |

## Files > 5 MB

none

## Data-like files outside ignored folders

none

## Secret scan (regex: cloud keys, GitHub/HF tokens, private keys, key=value secrets)

no matches

## Licence-restricted material (must stay out of git)

- data/raw/ (CARE-PD, gaitpdb, drawings), models/raw/ (Sabo checkpoints, SMPL, Parkonix/Swin/fastai weights), models/safe/ (our trained weights derived from CARE-PD)
- results/ keeps only small derived numbers/figures; per-walk CSVs contain CARE-PD walk IDs and labels (derived data) -> check CARE-PD terms before publishing them.

## Largest committed files

- results/target1/figures/A1_vs_A3_gradcam_spiral.png (1.76 MB)
- results/target1/figures/A3_SwinTiny_samples.png (1.44 MB)
- results/gait_motion/e0_carepd_external_per_walk.csv (1.35 MB)
- results/target1/figures/A1_MobileNet_TM_samples.png (1.34 MB)
- results/poc/injection_per_walk_fixedops.csv (0.93 MB)
- results/poc/injection_per_walk_lam002.csv (0.93 MB)
- results/poc/injection_per_walk.csv (0.93 MB)
- results/poc/injection_per_walk_nores.csv (0.93 MB)
- results/poc/blackbox_injection_per_walk_cnn_s1.csv (0.58 MB)
- results/poc/blackbox_injection_per_walk_cnn_s2.csv (0.57 MB)
- results/poc/blackbox_injection_per_walk_stgcn.csv (0.56 MB)
- results/poc/blackbox_injection_per_walk.csv (0.52 MB)
- results/poc/demo/poc_demo.mp4 (0.48 MB)
- paper/main.pdf (0.46 MB)
- results/gait_motion/animations/skeleton_3_3DGait_vid0078_0063b.mp4 (0.24 MB)
