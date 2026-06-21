#!/bin/bash
# 干净全局长治(global2, coverage_simplify-only 避 topojson OOM)启动器。
cd /home/ps/landform/sidecar
export PYTHONUNBUFFERED=1
~/miniconda3/bin/python run_changzhi_global2.py > /mnt/sda/zf/landform/results/changzhi_g2.log 2>&1
echo "G2_EXIT=$?" >> /mnt/sda/zf/landform/results/changzhi_g2.log
