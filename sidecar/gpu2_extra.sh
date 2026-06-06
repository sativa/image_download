#!/bin/bash
# Use the otherwise-idle GPU2 for a 5th ensemble member (unetplusplus arch),
# baseline then route-c, sequentially.
LF=/mnt/sda/zf/landform; SD=/home/ps/landform/sidecar; PY="$HOME/miniconda3/bin/python"
REG=/home/ps/landform/data/v40_5k.json; FEAT=$LF/data/c_stage1_feat
cd "$SD"; export CUDA_VISIBLE_DEVICES=2
echo "[gpu2] bl_pp (unet++ baseline) $(date '+%H:%M:%S')"
"$PY" -u train_c_stage2.py --regions-json "$REG" --no-1m --seed 0 --arch unetplusplus \
    --out-dir "$LF/results/ens_bl_pp" --device cuda:0 --workers 12 > "$LF/results/ens_bl_pp.log" 2>&1
echo "[gpu2] rc_pp (unet++ route-c) $(date '+%H:%M:%S')"
"$PY" -u train_c_stage2.py --regions-json "$REG" --feat-dir "$FEAT" --seed 0 --arch unetplusplus \
    --out-dir "$LF/results/ens_rc_pp" --device cuda:0 --workers 12 > "$LF/results/ens_rc_pp.log" 2>&1
echo "[gpu2] done $(date '+%H:%M:%S')"
