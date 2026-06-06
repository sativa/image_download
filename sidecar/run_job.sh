#!/bin/bash
# usage: run_job.sh SCRIPT.py GPU OUT "EXTRA ARGS..."
SCRIPT=$1; GPU=$2; OUT=$3; shift 3; EXTRA="$@"
cd /home/ps/landform/sidecar || exit 1
mkdir -p /mnt/sda/zf/landform/results
LOG=/mnt/sda/zf/landform/results/${OUT}.log
CUDA_VISIBLE_DEVICES=$GPU setsid ~/miniconda3/bin/python -u "$SCRIPT" \
  --device cuda:0 --out /mnt/sda/zf/landform/results/$OUT $EXTRA </dev/null >"$LOG" 2>&1 &
echo "launched pid $! gpu=$GPU script=$SCRIPT out=$OUT log=$LOG"
