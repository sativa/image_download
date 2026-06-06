#!/bin/bash
# z18 fine-tune pipeline: build z18 npz -> fine-tune z17 model on z18 -> eval z18-ft vs z17 baseline.
# Isolates RESOLUTION: same plain 6ch config + same start model (dino_1m), only the data changes (z17->z18).
# usage: z18_ft_pipeline.sh [GPU]
GPU=${1:-0}
R=/mnt/sda/zf/landform/results; D=/mnt/sda/zf/landform/data; PY=~/miniconda3/bin/python
cd /home/ps/landform/sidecar || exit 1

echo "===== [1/4] build z18 npz (esri+google+DLTB) ====="
$PY -u build_z18_npz.py --workers 24

echo "===== [2/4] fine-tune z17 model (dino_1m) on z18, plain 6ch, 12 ep ====="
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_dino_1m_v2.py --data-dir $D/c_1m_z18 \
  --warm-start $R/dino_1m/best.pt --epochs 12 --batch-size 6 --device cuda:0 --out $R/dino_z18_ft 2>&1 | tail -16

echo "===== [3/4] z18-FT model on z18 test (parcel-level) ====="
CUDA_VISIBLE_DEVICES=$GPU $PY -u parcel_eval.py --data-dir $D/c_1m_z18 --ckpt $R/dino_z18_ft/best.pt \
  --device cuda:0 2>&1 | grep -E "PIXEL|MMU|>="

echo "===== [4/4] z17 baseline (dino_1m) on z17 same cells (parcel-level) ====="
CUDA_VISIBLE_DEVICES=$GPU $PY -u parcel_eval.py --data-dir $D/c_1m --ckpt $R/dino_1m/best.pt --plain \
  --n-cells 120 --device cuda:0 2>&1 | grep -E "PIXEL|MMU|>="

echo "===== DONE: compare [3] z18-FT vs [4] z17 baseline; same plain 6ch config, resolution is the only change ====="
