#!/bin/bash
# Queue the v39 DINOv2-L multitemporal baseline (the paper needs this number).
# Waits for the v36 wave-1 seeds to finish (marker in the master log), then runs
# v39 on the freed GPU 1 -- concurrent with seed 4 on GPU 0, so GPU 1 isn't idle.
set -u
cd /home/ps/landform/sidecar
PY="$HOME/miniconda3/bin/python"
MASTER=/tmp/v36_ensemble_master.log
echo "[v39] waiting for v36 wave-1 to finish $(date '+%F %T')"
while ! grep -q 'wave1 complete' "$MASTER" 2>/dev/null; do sleep 60; done
echo "[v39] wave-1 done -> launching DINOv2-L baseline on GPU 1 $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=1 "$PY" -u train_v39_dino_multitemp.py \
  --device cuda:0 --out-dir /home/ps/landform/results/v39 \
  > /tmp/v39.log 2>&1
echo "[v39] done rc=$? $(date '+%F %T')"
