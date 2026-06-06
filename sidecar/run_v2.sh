#!/bin/bash
# usage: run_v2.sh GPU OUT "EXTRA ARGS..."
GPU=$1; OUT=$2; shift 2; EXTRA="$@"
cd /home/ps/landform/sidecar || exit 1
mkdir -p /mnt/sda/zf/landform/results
LOG=/mnt/sda/zf/landform/results/${OUT}.log
CUDA_VISIBLE_DEVICES=$GPU setsid ~/miniconda3/bin/python -u train_dino_1m_v2.py \
  --device cuda:0 --out /mnt/sda/zf/landform/results/$OUT $EXTRA </dev/null >"$LOG" 2>&1 &
echo "launched pid $! gpu=$GPU out=$OUT log=$LOG"
