#!/bin/bash
# 长治市全市 landuse polygon —— 干净全局法(正路,B)。
# 全局 by-construction 无缝(分幅有缝/重复已实证)。三道 city-scale 墙的对策:
#   ① mosaic 4GB → VRT 虚拟 mosaic(--mosaic);② 推理累加器内存 → infer_global_memmap(已并入,band-local+磁盘memmap,4卡不爆);
#   ③ topojson 内存(长治真 blocker)→ --tol 15 激进 coverage_simplify 减顶点。
# --save-intermediate 存 idmap.npy:topojson 若仍 OOM,可从 idmap 反复试更高 tol/巨斑简化,不重推理(~4h)。
cd /home/ps/landform/sidecar
export PYTHONUNBUFFERED=1
mkdir -p /mnt/sda/zf/landform/results/changzhi_inter
~/miniconda3/bin/python parcel_pipeline.py \
  --mosaic /mnt/sda/zf/landform/results/changzhi_mosaic.vrt \
  --weights /mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt \
  --backbone /home/ps/landform/dinov3/dinov3-vitl16-sat493m \
  --out /mnt/sda/zf/landform/results/changzhi_FINAL.parquet \
  --device cuda --gpus 0,1,2,3 \
  --boundary /mnt/sda/zf/landform/data/changzhi_boundary.geojson \
  --downscale 4 --smooth-iters 3 --tol 15 --utm EPSG:32649 \
  --save-intermediate /mnt/sda/zf/landform/results/changzhi_inter \
  > /mnt/sda/zf/landform/results/changzhi_FINAL.log 2>&1
echo "PIPELINE_EXIT=$?" >> /mnt/sda/zf/landform/results/changzhi_FINAL.log
