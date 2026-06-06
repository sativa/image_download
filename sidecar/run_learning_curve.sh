#!/bin/bash
# Learning curve A: baseline (10m-only) vs route-c (10m+1m), single unet seed0, at
# nested train subsets 1k/2.5k (5k already done = c_stage2_base / c_stage2_1m).
# Same 120 cross-county test cells. Trend -> extrapolate whether 20k would help + whether
# the 1m gain holds with more data. Zero extra download (reuses existing c_stage1_feat).
set -u
LF=/mnt/sda/zf/landform; SD=/home/ps/landform/sidecar; PY="$HOME/miniconda3/bin/python"
D=/home/ps/landform/data; FEAT=$LF/data/c_stage1_feat
cd "$SD"; echo "=== learning curve start $(date '+%F %T') ==="

# 4-GPU parallel: bl@1k(0) rc@1k(1) bl@2.5k(2) rc@2.5k(3)
nohup "$PY" -u train_c_stage2.py --regions-json $D/v40_1000.json --no-1m --seed 0 \
    --out-dir $LF/results/lc_bl_1k --device cuda:0 > $LF/results/lc_bl_1k.log 2>&1 &
nohup "$PY" -u train_c_stage2.py --regions-json $D/v40_1000.json --feat-dir $FEAT --seed 0 \
    --out-dir $LF/results/lc_rc_1k --device cuda:1 > $LF/results/lc_rc_1k.log 2>&1 &
nohup "$PY" -u train_c_stage2.py --regions-json $D/v40_2500.json --no-1m --seed 0 \
    --out-dir $LF/results/lc_bl_2k5 --device cuda:2 > $LF/results/lc_bl_2k5.log 2>&1 &
nohup "$PY" -u train_c_stage2.py --regions-json $D/v40_2500.json --feat-dir $FEAT --seed 0 \
    --out-dir $LF/results/lc_rc_2k5 --device cuda:3 > $LF/results/lc_rc_2k5.log 2>&1 &
wait; echo "[lc trainings done] $(date '+%T')"

cvf(){ grep -o '"cv_F1": [0-9.]*' "$1" 2>/dev/null | grep -o '[0-9.]*$'; }
b1=$(cvf $LF/results/lc_bl_1k/final.json);  r1=$(cvf $LF/results/lc_rc_1k/final.json)
b2=$(cvf $LF/results/lc_bl_2k5/final.json); r2=$(cvf $LF/results/lc_rc_2k5/final.json)
b5=$(cvf $LF/results/c_stage2_base/final.json); r5=$(cvf $LF/results/c_stage2_1m/final.json)
echo ""; echo "=== LEARNING CURVE (cross-county CV F1, single unet seed0) ==="
printf "  train=1.0k : baseline=%s  route-c=%s\n" "$b1" "$r1"
printf "  train=2.5k : baseline=%s  route-c=%s\n" "$b2" "$r2"
printf "  train=5.0k : baseline=%s  route-c=%s\n" "$b5" "$r5"
printf "  train=20k  : baseline=0.853 (ensemble ref)  route-c=?(=option B, needs 1m@20k)\n"
echo "=== learning curve end $(date '+%F %T') ==="
