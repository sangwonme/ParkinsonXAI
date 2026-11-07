# ParkinsonXAI

Research code for **"Saliency maps do not explain Parkinson's gait models — a symptom-operator model that does"**
(proof of concept, November 2026).

Two questions:
1. Do the explanations (Grad-CAM, Integrated Gradients, occlusion) of accurate deep Parkinson's disease (PD)
   detectors actually describe what the models use? We test public pretrained models on hand drawings and on
   gait (2D pose), with sanity checks, deletion tests, shortcut tests and synthetic symptom injection.
2. Can a gait model be accurate enough while being explainable *by construction*? We propose a model whose
   prediction is a sum of contributions of measurable gait symptoms (stride, arm swing, asymmetry, ...) plus a
   small penalised residual, and compare it with a black-box CNN and with classic machine learning on the same
   gait features.

Start with `RESEARCH_CONTEXT.md` (full story, decisions, results, open questions). Agents: read `AGENTS.md`.
Paper draft: `paper/main.pdf`. Demo video: `results/poc/demo/poc_demo.mp4`.

## Repository layout (short)
```
src/xaipd/        library code (data loaders, models, explainers, symptom operators, safety helpers)
scripts/          one script per experiment (see RESEARCH_CONTEXT.md "File map")
results/          small result files (json/csv/md/png/mp4); large arrays are git-ignored
paper/            LaTeX source + figures + compiled PDF
docs/             literature check
PLAN.md, NOTES.md lab notebook (Korean)
```

## Setup (Windows 11, NVIDIA GPU optional)
```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/Scripts/python.exe -r envs/requirements-lock.txt   # numpy must stay < 2
```
Every script imports `xaipd.memguard` first (6 GB memory guard, low CPU priority, `XAIPD_THREADS` threads).
Run headless: `MPLBACKEND=Agg`.

## Getting the data and weights (not included: non-commercial licences)
| item | where to get it | licence | put it in |
|---|---|---|---|
| CARE-PD gait (SMPL), `*_canonical.pkl` + `*_6fold_participants.pkl` for 3DGait, BMCLab, PD-GaM, T-SDU-PD | Dataverse doi:10.5683/SP3/TWIKMK (or HF `vida-adl/CARE-PD`); read the terms at neurips2025.care-pd.ca | CC BY-NC / BY-NC-ND 4.0 (metadata disagree) | `data/raw/care_pd/` |
| SMPL neutral body model `SMPL_NEUTRAL.pkl` (used only for rest joints) | CARE-PD GitHub repo `data/preprocessing/common/body_models/smpl/` or smpl.is.tue.mpg.de | SMPL non-commercial | `models/raw/smpl/` |
| Sabo et al. ST-GCN checkpoints + 15 labelled sample walks | github.com/TaatiTeam/stgcn_parkinsonism_prediction (`sample_data/`) | UHN academic, non-commercial | `models/raw/sabo_stgcn/` |
| Hand drawings (spiral/wave, 204 images) | github.com/SaiJeevanPuchakayala/Parkonix (`Data/drawings`), same images as Kaggle kmader/parkinsons-drawings | CC BY-NC-ND 4.0 (stated on a model card) | `data/raw/parkonix_repo/` |
| Drawing models: Parkonix `keras_model.h5`, HF `gianlab/swin-tiny-...-finetuned-parkinson-classification` | GitHub / Hugging Face | not stated / Apache-2.0 | `models/raw/a1_parkonix/`, `models/raw/a3_gianlab_swin/` |

Checksums of what we used: `models/CHECKSUMS.txt`, `models/raw/sabo_CHECKSUMS.txt` (local), `data/raw/care_pd/SHA256.txt`
(local), CARE-PD MD5 list in `NOTES.md`. Pickled files are opened only through `src/xaipd/safe_pickle.py`.

## Reproduce the PoC (CPU is enough except the black-box CNN, ~3 min per fold on CPU)
```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); from xaipd import symptoms as S; S.build_cache()"
.venv/Scripts/python.exe scripts/poc_symptom.py 40      # proposed model + logistic regression + random forest
.venv/Scripts/python.exe scripts/poc_blackbox.py 12     # black-box CNN + Grad-CAM / IG (needs splits from the line above)
.venv/Scripts/python.exe scripts/poc_report.py          # tables + paper figures
.venv/Scripts/python.exe scripts/poc_demo_video.py      # demo video
```
Earlier analyses (drawings, Sabo ST-GCN): `scripts/run_target1.py`, `images_e5_e6.py`, `sabo_xai.py`, `sabo_followup.py`,
`sabo_skeleton_anim.py`.

## Status
Proof of concept, single machine, not peer reviewed. Clinical use is not intended.
