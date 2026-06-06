#!/bin/bash
# Orchestrate the terrace-6000 expansion retrain: wait for c_1m_plus2 (10500), launch 3 configs across
# GPUs, then dual-protocol eval (standard 120-cell test + held-out 500 terrace test). Server-side setsid
# so it survives ssh drops; writes results to $R/terrace_retrain_RESULTS.txt and a DONE marker.
D=/mnt/sda/zf/landform/data
R=/mnt/sda/zf/landform/results
SC=/home/ps/landform/sidecar
PY=/home/ps/miniconda3/bin/python
BK=/home/ps/landform/dinov3/dinov3-vitl16-sat493m
RES=$R/terrace_retrain_RESULTS.txt
cd $SC
echo "[orch] waiting for c_1m_plus2 ..." >$RES

until [ -f $D/c_1m_plus2/manifest.json ]; do sleep 30; done
echo "[orch] c_1m_plus2 ready: $($PY -c "import json;m=json.load(open('$D/c_1m_plus2/manifest.json'));print('train',len(m['train']),'test',len(m['test']))")" >>$RES

COMMON="--multitemporal --boundary-head --small-weight 4 --small-k 31 --epochs 20"
# GPU0: GDLX (primary)
CUDA_VISIBLE_DEVICES=0 setsid $PY -u train_dino_1m_v3.py --device cuda:0 --out $R/dino_1m_v3_gdlx_t2 \
  --data-dir $D/c_1m_plus2 $COMMON --gdlx-head --gdlx-weight 0.3 --batch-size 6 \
  </dev/null >$R/dino_1m_v3_gdlx_t2.log 2>&1 &
# GPU1: GDLX + FreqFusion
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid $PY -u train_dino_1m_v3.py --device cuda:0 --out $R/dino_1m_v3_gdlxff_t2 \
  --data-dir $D/c_1m_plus2 $COMMON --gdlx-head --gdlx-weight 0.3 --freqfusion --batch-size 3 \
  </dev/null >$R/dino_1m_v3_gdlxff_t2.log 2>&1 &
# GPU2: plain DINOv3 (data-scaling baseline)
CUDA_VISIBLE_DEVICES=2 setsid $PY -u train_dino_1m_v3.py --device cuda:0 --out $R/dino_1m_v3_plain_t2 \
  --data-dir $D/c_1m_plus2 $COMMON --batch-size 6 \
  </dev/null >$R/dino_1m_v3_plain_t2.log 2>&1 &
echo "[orch] launched 3 trainings on GPU0/1/2" >>$RES

until grep -q "ep20/20" $R/dino_1m_v3_gdlx_t2.log 2>/dev/null \
   && grep -q "ep20/20" $R/dino_1m_v3_gdlxff_t2.log 2>/dev/null \
   && grep -q "ep20/20" $R/dino_1m_v3_plain_t2.log 2>/dev/null; do sleep 60; done
echo "[orch] all 3 trainings done; evaluating" >>$RES

# eval: $1=tag $2=extra-eval-flags
evalone () {
  local tag=$1; shift; local flags="$@"
  echo "" >>$RES; echo "===== $tag =====" >>$RES
  echo "  best: $(grep best= $R/dino_1m_v3_${tag}.log | tail -1)" >>$RES
  echo "  --- STANDARD test (120 cells, c_1m_plus2) ---" >>$RES
  CUDA_VISIBLE_DEVICES=0 $PY -u parcel_eval.py --ckpt $R/dino_1m_v3_${tag}/best.pt \
    --v3-backbone $BK $flags --multitemporal --data-dir $D/c_1m_plus2 --n-cells 120 2>&1 \
    | grep ">=0.0\|>=0.05\|>=0.5\|PIXEL" >>$RES
  echo "  --- TERRACE held-out test (500 cells, c_1m_terrace2) ---" >>$RES
  CUDA_VISIBLE_DEVICES=0 $PY -u parcel_eval.py --ckpt $R/dino_1m_v3_${tag}/best.pt \
    --v3-backbone $BK $flags --multitemporal --data-dir $D/c_1m_terrace2 --n-cells 500 2>&1 \
    | grep ">=0.0\|>=0.05\|>=0.5\|PIXEL" >>$RES
}
evalone gdlx_t2
evalone gdlxff_t2 --v3-freq
evalone plain_t2
echo "" >>$RES
echo "=== 基线对比: GDLX 5000 小地块0.751/面积0.933 · FreqFusion 0.749/0.935 · plain~0.745 ===" >>$RES
echo "ORCH_DONE" >>$RES
