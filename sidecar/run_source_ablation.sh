#!/bin/bash
# Dual-vs-single source ablation (Gansu cross-county). Dual (both, 6ch) is already done
# (c_stage1 / c_stage1_feat / c_stage2_1m). Train esri-only + google-only (3ch each),
# all else identical (same cells / seed0 unet / recipe), and compare stage-2 CV F1.
set -u
LF=/mnt/sda/zf/landform; SD=/home/ps/landform/sidecar; PY="$HOME/miniconda3/bin/python"
REG=/home/ps/landform/data/v40_5k.json; S2=/home/ps/landform/data/v19_s2_raw
cd "$SD"; echo "=== source ablation start $(date '+%F %T') ==="

# 1) stage-1: esri (GPU0) || google (GPU1)
nohup "$PY" -u train_c_stage1.py --data-dir "$LF/data/c_1m" --sources esri \
    --out "$LF/results/c_stage1_esri" --device cuda:0 > "$LF/results/s1_esri.log" 2>&1 &
P1=$!
nohup "$PY" -u train_c_stage1.py --data-dir "$LF/data/c_1m" --sources google \
    --out "$LF/results/c_stage1_google" --device cuda:1 > "$LF/results/s1_google.log" 2>&1 &
P2=$!
echo "stage-1 esri pid=$P1 (GPU0) | google pid=$P2 (GPU1)"; wait $P1 $P2
echo "[stage-1 done] $(date '+%T')"

# 2) stage-1 inference -> 2-ch feats (esri GPU0 || google GPU1)
nohup "$PY" -u stage1_infer.py --data-dir "$LF/data/c_1m" --s2-dir "$S2" --sources esri \
    --ckpt "$LF/results/c_stage1_esri/best.pt" --out-dir "$LF/data/c_stage1_feat_esri" \
    --device cuda:0 > "$LF/results/s1infer_esri.log" 2>&1 &
I1=$!
nohup "$PY" -u stage1_infer.py --data-dir "$LF/data/c_1m" --s2-dir "$S2" --sources google \
    --ckpt "$LF/results/c_stage1_google/best.pt" --out-dir "$LF/data/c_stage1_feat_google" \
    --device cuda:1 > "$LF/results/s1infer_google.log" 2>&1 &
I2=$!
wait $I1 $I2; echo "[infer done] $(date '+%T')"

# 3) stage-2 (seed0 unet, same as dual c_stage2_1m): esri (GPU2) || google (GPU3)
nohup "$PY" -u train_c_stage2.py --regions-json "$REG" --feat-dir "$LF/data/c_stage1_feat_esri" \
    --seed 0 --out-dir "$LF/results/c_stage2_esri" --device cuda:2 > "$LF/results/s2_esri.log" 2>&1 &
A=$!
nohup "$PY" -u train_c_stage2.py --regions-json "$REG" --feat-dir "$LF/data/c_stage1_feat_google" \
    --seed 0 --out-dir "$LF/results/c_stage2_google" --device cuda:3 > "$LF/results/s2_google.log" 2>&1 &
B=$!
wait $A $B; echo "[stage-2 done] $(date '+%T')"

cvf(){ grep -o '"cv_F1": [0-9.]*' "$1" 2>/dev/null | grep -o '[0-9.]*$'; }
s1f(){ grep -o 'best 1m-F1=[0-9.]*' "$1" 2>/dev/null | grep -o '[0-9.]*$'; }
echo ""; echo "=== SOURCE ABLATION (Gansu cross-county, single unet seed0) ==="
echo "stage-1 1m-F1:  dual=$(s1f $LF/results/s1.log)  esri=$(s1f $LF/results/s1_esri.log)  google=$(s1f $LF/results/s1_google.log)"
echo "stage-2 CV F1:  dual(6ch)=$(cvf $LF/results/c_stage2_1m/final.json)  esri(3ch)=$(cvf $LF/results/c_stage2_esri/final.json)  google(3ch)=$(cvf $LF/results/c_stage2_google/final.json)"
echo "baseline(10m,no-1m)=$(cvf $LF/results/c_stage2_base/final.json)"
echo "=== source ablation end $(date '+%F %T') ==="
