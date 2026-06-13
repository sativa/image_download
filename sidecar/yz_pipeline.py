"""yz_pipeline — 榆中(620123)示例: 调用通用 parcel_pipeline 的 thin wrapper.

把榆中的具体参数(mosaic 路径、四头权重、DINOv3-Sat backbone、县界 parquet、降采样、平滑)
喂给区域无关的 `parcel_pipeline.run_pipeline`。一切算法在通用模块里, 这里只放榆中常量。

新区域照抄此文件改 5 个常量即可(或直接命令行跑 parcel_pipeline.py)。历史 yz_global_ffl/
yz_smooth2/yz_postproc 保留不动(它们是原始分阶段脚本 + 帧场实验, 通用版默认不用 FFL)。

Run on .250:
  /home/ps/miniconda3/bin/python yz_pipeline.py
"""
import sys
from pathlib import Path

SIDECAR = str(Path(__file__).resolve().parent)
if SIDECAR not in sys.path:
    sys.path.insert(0, SIDECAR)
import parcel_pipeline as pp

# ---- 榆中(620123)常量 ----
MOSAIC = "/mnt/sda/zf/landform/results/yuzhong_county_mosaic.tif"
WEIGHTS = "/mnt/sda/zf/landform/results/dino_v3_bddf/last.pt"   # 四头 BDDF(cls9+bnd+dist+frame)
BACKBONE = "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"
BOUNDARY = "/tmp/yz_county_boundary.parquet"                    # 榆中县界(从 DLTB 620123 union 出)
OUT = "/mnt/sda/zf/landform/results/yuzhong_pipeline.parquet"
DOWNSCALE = 4
SMOOTH_ITERS = 2
UTM = "EPSG:32648"          # 甘肃榆中 48N


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Yuzhong 620123 example wrapper over parcel_pipeline")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mosaic", default=MOSAIC)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--boundary", default=BOUNDARY)
    a = ap.parse_args()
    pp.run_pipeline(a.mosaic, a.weights, BACKBONE, a.out, boundary=a.boundary,
                    downscale=DOWNSCALE, smooth_iters=SMOOTH_ITERS, classes=None,
                    gpus=a.gpus.split(","), utm=UTM)


if __name__ == "__main__":
    main()
