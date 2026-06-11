#!/bin/bash
# 榆中端到端隔夜编排:等梯田enhance6微调完成 → 用新权重跑县级(enhance+Hann+dist-peak+halo+shapes)
# → 分幅缝合+平滑+裁县界+出图。全程 setsid 后台,产物在 results/yuzhong_FINAL*。
set -u
cd /home/ps/landform/sidecar
PY=/home/ps/miniconda3/bin/python
R=/mnt/sda/zf/landform/results
LOG=$R/orchestrator.log
echo "[orch] $(date) waiting for training dino_v3_bddf_enh ..." > $LOG
# 1) 等训练完成:best.pt 存在 且 训练进程已退出
while pgrep -f "train_dino_7class.*dino_v3_bddf_enh" >/dev/null || [ ! -f $R/dino_v3_bddf_enh/best.pt ]; do
  sleep 120
done
sleep 30
echo "[orch] $(date) training done." >> $LOG
grep -E "FINAL|best=" $R/bddf_enh.log 2>/dev/null | tail -3 >> $LOG
ls -la $R/dino_v3_bddf_enh/best.pt >> $LOG 2>&1

# 2) 县级重做:新权重 + enhance6 + 6x6 halo + dist-peak(downscale) + shapes(精确)
echo "[orch] $(date) running yz_blocks full county (enhanced) ..." >> $LOG
rm -rf $R/yz_enh $R/yuzhong_enh_region.parquet
$PY yz_blocks.py --block 6 --halo-cells 2 --enhance \
  --weights $R/dino_v3_bddf_enh/best.pt \
  --gpus 0,1,2,3 --per-gpu 1 \
  --out-dir $R/yz_enh --county-out $R/yuzhong_enh_region.parquet >> $LOG 2>&1
echo "[orch] $(date) county delineation done." >> $LOG

# 3) 分幅缝合 + 平滑 + 裁县界 + 出图 + 对账
echo "[orch] $(date) finalizing (分幅缝合/平滑/裁界/出图) ..." >> $LOG
$PY seam_finalize.py --in $R/yuzhong_enh_region.parquet --out $R/yuzhong_FINAL.parquet --tag FINAL >> $LOG 2>&1

echo "[orch] $(date) ALL_DONE" >> $LOG
touch $R/ORCHESTRATOR_DONE
