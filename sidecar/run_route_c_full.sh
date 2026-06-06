#!/bin/bash
# Route-c full pipeline on .250 (run after fuse produces c_1m). 4-GPU parallel.
set -u
LF=/mnt/sda/zf/landform
SD=/home/ps/landform/sidecar
PY="$HOME/miniconda3/bin/python"
REG=/home/ps/landform/data/v40_5k.json
S2=/home/ps/landform/data/v19_s2_raw
mkdir -p "$LF/results"; cd "$SD"
echo "=== route-c full start $(date '+%F %T') ==="

# 0) coverage: how many of the 120 cross-county test cells actually got 1m
"$PY" - <<PY
import json
m=json.load(open("$LF/data/c_1m/manifest.json")); R=json.load(open("$REG"))
test=set(f"{c['county']}_{c['idx']}" for c in R["test"])
print(f"[cov] c_1m train={len(m['train'])} test={len(m['test'])}; of 120 test cells {len(test & set(m['test']))} have 1m")
PY

# 1) stage-1 (1m, GPU0)  ||  stage-2 10m-only baseline (GPU1) — fully parallel
nohup "$PY" -u train_c_stage1.py --data-dir "$LF/data/c_1m" --out "$LF/results/c_stage1" \
    --device cuda:0 > "$LF/results/s1.log" 2>&1 &
P1=$!
nohup "$PY" -u train_c_stage2.py --regions-json "$REG" --no-1m --seed 0 \
    --out-dir "$LF/results/c_stage2_base" --device cuda:1 > "$LF/results/s2_base.log" 2>&1 &
P2=$!
echo "launched: stage-1 pid=$P1 (GPU0) | stage-2 baseline pid=$P2 (GPU1)"

wait $P1; echo "[stage-1 done] $(date '+%T')"; tail -2 "$LF/results/s1.log"

# 2) stage-1 inference -> 2-ch features at 10m (GPU0, tiled)
"$PY" -u stage1_infer.py --data-dir "$LF/data/c_1m" --s2-dir "$S2" \
    --ckpt "$LF/results/c_stage1/best.pt" --out-dir "$LF/data/c_stage1_feat" \
    --device cuda:0 > "$LF/results/s1_infer.log" 2>&1
echo "[infer done] $(date '+%T')"; tail -2 "$LF/results/s1_infer.log"

# 3) stage-2 route-c (10m + 1m, GPU2)
"$PY" -u train_c_stage2.py --regions-json "$REG" --feat-dir "$LF/data/c_stage1_feat" --seed 0 \
    --out-dir "$LF/results/c_stage2_1m" --device cuda:2 > "$LF/results/s2_1m.log" 2>&1
echo "[stage-2 route-c done] $(date '+%T')"

wait $P2 2>/dev/null
echo ""; echo "=== COMPARISON (cross-county F1, 120 held-out cells / 12 counties) ==="
grep -h "\[FINAL\]" "$LF/results/s2_base.log" "$LF/results/s2_1m.log"
echo "reference: 10m ENSEMBLE @20k-train cross-county F1 = 0.853 (acc 0.906)"
echo "=== route-c full end $(date '+%F %T') ==="
