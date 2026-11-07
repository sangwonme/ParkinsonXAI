#!/usr/bin/env bash
# Parkinson GPU queue (coordinated with GenerativeUI).
# Start condition: GenerativeUI's queue log contains "GPU YIELD to Parkinson" AND C:\Works\KSEF26\.gpu_busy is absent.
# While running we own C:\Works\KSEF26\.gpu_busy ("parkinson <time>"). On exit (success, error, kill) the trap removes
# our lock (only if it is ours) and writes C:\Works\KSEF26\.gpu_parkinson_done with the end time.
# Budget: GenerativeUI yields at most 45 min -> hard deadline 38 min for all jobs (each job also time-limited).
set -u
LOCK=/c/Works/KSEF26/.gpu_busy
DONE=/c/Works/KSEF26/.gpu_parkinson_done
YIELD_LOG=/c/Works/KSEF26/GenerativeUI/runs/poc/queue.log
cd /c/Works/KSEF26/XAIParkinson
LOG=results/poc/gpu_queue.log
ts() { date "+%Y-%m-%d %H:%M:%S"; }
OWN=0

cleanup() {
  if [ "$OWN" = 1 ] && [ -f "$LOCK" ] && grep -q "^parkinson" "$LOCK"; then rm -f "$LOCK"; fi
  if [ "$OWN" = 1 ]; then echo "$(ts) parkinson GPU jobs finished" > "$DONE"; fi
  echo "$(ts) queue exit (own=$OWN)" >> "$LOG"
}
trap cleanup EXIT INT TERM

echo "$(ts) waiting for 'GPU YIELD to Parkinson' in $YIELD_LOG" >> "$LOG"
until [ -f "$YIELD_LOG" ] && grep -q "GPU YIELD to Parkinson" "$YIELD_LOG"; do sleep 15; done
while [ -f "$LOCK" ]; do sleep 5; done
echo "parkinson $(ts)" > "$LOCK"; OWN=1
START=$(date +%s); DEADLINE=$((START + 38 * 60))
echo "$(ts) GPU acquired; $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$LOG"

run() {  # run <name> <max_seconds> <env...> -- <args>
  local name=$1 maxs=$2; shift 2
  local left=$((DEADLINE - $(date +%s)))
  if [ "$left" -lt 120 ]; then echo "$(ts) skip $name (budget exhausted)" >> "$LOG"; return; fi
  [ "$maxs" -gt "$left" ] && maxs=$left
  echo "$(ts) start $name (limit ${maxs}s)" >> "$LOG"
  env "$@" timeout "$maxs" .venv/Scripts/python.exe scripts/poc_blackbox.py "${EPOCHS}" > "results/poc/blackbox${TAGNAME}.log" 2>&1
  echo "$(ts) end $name exit=$?" >> "$LOG"
}
EPOCHS=12; TAGNAME=_cnn_s1; run cnn_seed1 600 BB_SEED=1 BB_TAG=_cnn_s1
EPOCHS=12; TAGNAME=_cnn_s2; run cnn_seed2 600 BB_SEED=2 BB_TAG=_cnn_s2
EPOCHS=10; TAGNAME=_stgcn;  run stgcn 1080 BB_ARCH=stgcn BB_TAG=_stgcn
exit 0
