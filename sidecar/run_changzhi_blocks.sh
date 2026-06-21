#!/bin/bash
# 长治市全市 landuse polygon —— 分幅法(yz_blocks 方案B 质心归属),全局法在 topojson 撞 OOM 后的正路。
# 每块独立 infer→idmap→vectorize→smooth,质心在 core 的地块整块吐出(halo 保证看全)→ 真无缝、内存随块不随城市。
# 配 dino_v3_bddf_enh/best.pt + --enhance(bddf_enh 训练含 enhance6)+ --ridge(田块级)+ --centroid(无缝)。
# 冒烟块 b11_00 已存在会被跳过(可续传)。
cd /home/ps/landform/sidecar
export PYTHONUNBUFFERED=1
~/miniconda3/bin/python yz_blocks.py \
  --regions /mnt/sda/zf/landform/data/changzhi_full_regions.json \
  --tif-dir /mnt/sda/zf/landform/data/changzhi_cells \
  --out-dir /mnt/sda/zf/landform/results/changzhi_blocks \
  --county-out /mnt/sda/zf/landform/results/changzhi_FINAL.parquet \
  --weights /mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt \
  --backbone /home/ps/landform/dinov3/dinov3-vitl16-sat493m \
  --gpus 0,1,2,3 --per-gpu 2 --block 6 --halo-cells 2 --centroid --enhance --ridge \
  > /mnt/sda/zf/landform/results/changzhi_blocks.log 2>&1
echo "BLOCKS_EXIT=$?" >> /mnt/sda/zf/landform/results/changzhi_blocks.log
