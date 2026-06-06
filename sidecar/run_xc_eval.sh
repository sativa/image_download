#!/bin/bash
# Auto-run the full evaluation matrix once the cross-county ensemble finishes.
#   Gansu cross-county : argmax + leave-county-out CV threshold (+ D4 TTA)
#   Changzhi x-province: argmax + Gansu-derived threshold TRANSFERRED (+ D4 TTA)
# All output -> /tmp/xc_eval_run.log
set -u
cd /home/ps/landform/sidecar
PY="$HOME/miniconda3/bin/python"
DATA=/home/ps/landform/data
RES=/home/ps/landform/results
exec > /tmp/xc_eval_run.log 2>&1

echo "[eval] waiting for ensemble training [ALL DONE] ..."
while ! grep -q "ALL DONE" /tmp/xc_master.log 2>/dev/null; do sleep 120; done
echo "[eval] training finished -> full eval matrix $(date '+%F %T')"
ls -1 $RES/xc_*/best.pt 2>/dev/null

run(){ CUDA_VISIBLE_DEVICES=0 "$PY" -u eval_xcounty.py "$@" --device cuda:0 2>&1 \
        | grep -E "test:|F1=|ENS|diag|wrote"; }

echo; echo "===== GANSU cross-county (argmax + CV-threshold) ====="
run --member-set xc --tag gansu
echo; echo "===== GANSU cross-county (+ D4 TTA) ====="
run --member-set xc --tta --tag gansu

# Transfer the Gansu 'all'-ensemble F1-optimal threshold to the cross-province test
TSTAR=$("$PY" -c "import json;print(json.load(open('$RES/xc_eval_gansu_argmax.json'))['ensembles'].get('all',{}).get('global_best_t',0.3))" 2>/dev/null || echo 0.3)
echo; echo "===== transfer threshold from Gansu: T*=$TSTAR ====="
echo; echo "===== CHANGZHI cross-province (argmax + Gansu T*) ====="
run --member-set xc --cells-pkl $DATA/changzhi_cells.pkl --fixed-threshold "$TSTAR" --tag changzhi
echo; echo "===== CHANGZHI cross-province (+ D4 TTA + Gansu T*) ====="
run --member-set xc --cells-pkl $DATA/changzhi_cells.pkl --fixed-threshold "$TSTAR" --tta --tag changzhi
echo; echo "[eval] ALL EVAL DONE $(date '+%F %T')"
