#!/bin/bash
# Cross-county DIVERSE ensemble on the 77/12 county-disjoint split (v40).
#   Train = 77 counties (19,753 cells); test = 12 held-out counties (120 cells).
#   Members: 4x EfficientNet-B5 U-Net (CE seeds 0-3) + UNet++ + DeepLabV3+ +
#            SegFormer (dice_ce, arch diversity) + v39 DINOv2 baseline.
# SEG_EW=none if SegFormer's mit_b5 ImageNet weights are unreachable on the box.
set -u
cd /home/ps/landform/sidecar
PY="$HOME/miniconda3/bin/python"
REG=/home/ps/landform/data/v40_xcounty_regions.json
RES=/home/ps/landform/results
SEG_EW="${SEG_EW:-imagenet}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # box has no direct huggingface.co; mit_b5 via mirror

run(){  # name gpu arch backbone loss seed [extra args...]
  local name=$1 gpu=$2 arch=$3 bk=$4 loss=$5 seed=$6; shift 6
  echo "[launch] $name gpu=$gpu arch=$arch bk=$bk loss=$loss seed=$seed $(date '+%T')"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u train_v33_multitemporal.py \
    --regions-json "$REG" --arch "$arch" --backbone "$bk" --loss "$loss" --seed "$seed" \
    --out-dir "$RES/$name" --device cuda:0 "$@" > "/tmp/$name.log" 2>&1
  echo "[done]   $name rc=$? $(date '+%T')"
}

echo "[xc-ensemble] start $(date '+%F %T')  SEG_EW=$SEG_EW"
# ---- Wave 1: cached-weight CNN members (zero download risk) ----
run xc_b5unet_s0 0 unet          efficientnet-b5 ce      0 & sleep 15
run xc_b5unet_s1 1 unet          efficientnet-b5 ce      1 & sleep 15
run xc_unetpp    2 unetplusplus  efficientnet-b5 dice_ce 0 & sleep 15
run xc_deeplab   3 deeplabv3plus efficientnet-b5 dice_ce 0 &
wait
echo "[wave1 done] $(date '+%T')"
# ---- Wave 2: more seeds + SegFormer (transformer) + DINOv2 baseline ----
run xc_b5unet_s2 0 unet          efficientnet-b5 ce      2 & sleep 15
run xc_b5unet_s3 1 unet          efficientnet-b5 ce      3 & sleep 15
run xc_segformer 2 segformer     mit_b5          dice_ce 0 --encoder-weights "$SEG_EW" & sleep 15
( echo "[launch] xc_v39 gpu=3 $(date '+%T')"
  CUDA_VISIBLE_DEVICES=3 "$PY" -u train_v39_dino_multitemp.py \
    --regions-json "$REG" --out-dir "$RES/xc_v39" --device cuda:0 > /tmp/xc_v39.log 2>&1
  echo "[done]   xc_v39 rc=$? $(date '+%T')" ) &
wait
echo "[ALL DONE] $(date '+%F %T')"
