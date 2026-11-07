# AGENTS.md — operating rules for AI coding agents on this repository

Read `RESEARCH_CONTEXT.md` first (goal, timeline, decisions, results, open questions), then `README.md`
(setup and data). `PLAN.md` and `NOTES.md` are the running lab notebook (Korean); append to them, do not rewrite history.

## Hard rules
1. **Never redistribute data or third-party weights.** `data/raw/`, `models/raw/`, `models/safe/` are git-ignored.
   CARE-PD (CC BY-NC / BY-NC-ND 4.0), Sabo et al. ST-GCN weights (UHN academic non-commercial), the SMPL body
   model (non-commercial) and the Kaggle drawing set (CC BY-NC-ND) must stay out of git. Re-download with README steps.
2. **Downloaded files are untrusted.**
   - Put them under `data/raw/<name>/` or `models/raw/<name>/`; never run scripts that live inside downloads.
   - PyTorch checkpoints: `torch.load(..., weights_only=True)` only. If that fails, stop and report; never set False.
   - Pickled data (CARE-PD `.pkl`, SMPL model): only via `src/xaipd/safe_pickle.py` (static opcode scan + allow-list
     unpickler; chumpy objects become inert stubs). Never `pickle.load` directly.
   - Keras `.h5`: inspect `model_config` with `scripts/inspect_h5.py` before loading (no `Lambda`/custom layers).
   - Prefer `safetensors` for anything we save.
3. **Resource limits** (the machine is shared; the owner may be gaming or measuring GPU latency):
   - import `xaipd.memguard` first in every script: process aborts above 6 GB RSS, BelowNormal priority,
     `XAIPD_THREADS` CPU threads (default 4; use 2 when another project is benchmarking).
   - GPU only when `C:\Works\KSEF26\.gpu_busy` does not exist (another project creates it while timing the GPU).
   - Process images / walks one at a time for memory-heavy explainers (Score-CAM once committed 51 GB when batched).
4. **Headless only:** `MPLBACKEND=Agg`; never open windows, viewers, browsers or installers.
5. **Honesty in reporting:** separate verified results from hypotheses; mark guesses as guesses; keep the
   "reproduced training" vs "public weights" distinction; report failures and negative results.
6. Do not commit or push unless the human owner explicitly asks.

## Environment
- Windows 11, RTX 4070 12 GB, Python 3.11 venv in `.venv` (created with `uv venv --python 3.11`).
- Lock file: `envs/requirements-lock.txt` (`uv pip install --python .venv/Scripts/python.exe -r envs/requirements-lock.txt`;
  torch comes from the `https://download.pytorch.org/whl/cu126` index). numpy must stay < 2 (TensorFlow 2.15 for the Keras drawing model).

## Where things are
See the file map in `RESEARCH_CONTEXT.md`.
