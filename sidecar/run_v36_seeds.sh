#!/bin/bash
# 5-seed v36 deep ensemble.
#   v36 == train_v33_multitemporal.py with --backbone efficientnet-b5 (9-ch multitemporal).
#   Each seed -> its own out-dir results/v36_s{N}/best.pt, logged to /tmp/v36_s{N}.log.
# Schedule on the 4x RTX 4090 box (PS4090):
#   wave 1: seeds 0,1,2,3 in parallel, one per GPU (staggered 20s to ease cold-cache disk I/O)
#   wave 2: seed 4 on GPU 0 after wave 1 frees it
set -u
cd /home/ps/landform/sidecar
PY="$HOME/miniconda3/bin/python"
SCRIPT=train_v33_multitemporal.py
RESULTS=/home/ps/landform/results

run_seed() {  # $1=seed  $2=gpu
  local seed=$1 gpu=$2
  echo "[launch] seed=$seed gpu=$gpu $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$SCRIPT" \
    --backbone efficientnet-b5 --seed "$seed" \
    --out-dir "$RESULTS/v36_s$seed" --device cuda:0 \
    > "/tmp/v36_s$seed.log" 2>&1
  echo "[done]   seed=$seed gpu=$gpu rc=$? $(date '+%F %T')"
}

echo "[ensemble] start $(date '+%F %T')"
for s in 0 1 2 3; do run_seed "$s" "$s" & sleep 20; done
wait
echo "[wave1 complete] $(date '+%F %T')"
run_seed 4 0
echo "[ALL DONE] $(date '+%F %T')"
