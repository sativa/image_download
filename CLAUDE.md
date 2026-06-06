# CLAUDE.md — imagery_downloader（1m 影像下载器 + 耕地分类 ML sidecar）

## 这个项目是什么
两部分:
1. **Tauri 2 桌面/CLI 影像下载器**(`src-tauri/` Rust + `src/` Svelte 前端):下 XYZ 瓦片(Esri/Google),拼接 + 裁剪到 bbox,
   写 EPSG:3857 GeoTIFF。有无头 CLI:`imagery-downloader batch`。
2. **Python ML sidecar**(`sidecar/`):用下载的影像做**耕地(耕地+园地)二分类**研究,目标是把 **1m 高分**接进来提升识别。

## ⚠️ 当前研究方向(最重要,别走偏)
**1m 高分是主体,10m 光谱是辅助。** 正确架构 = **1m 地块/地貌分割(SAM3 / DINO 微调)为主 → 逐地块(OBIA)聚合
10m 光谱+多年 NDVI → 判类(耕地/园地/其他)**。**绝不能让 10m 反客为主**(之前的 route-c 就犯了这个错:1m 降采样当提示、
在 10m 预测——已废弃方向)。在 **1m、地块级**评估。
- **DLTB(三调)= 权威标准真值**,当作准,不要说成"标签噪声/对齐误差"。
- 评估必报 **甘肃跨县 + 山西长治跨省** 两个口径;跨省必报 argmax(部署点)和 best-thr(上界)。

## 关键研究结论(详见 `sidecar/1M_FUSION_LESSONS.md` + 记忆 cropland-seg-honest-eval)
- 纯 10m 集成跨县 F1 0.853 / 准确率 0.906;**1m 同域提升有限(+0.009)、~0.86 封顶、20k 后归零、单源够**。
- **1m 的真价值在跨域**:纯 10m 跨省崩到 argmax 0.236,加 1m 救到 0.644(+0.41);跨省双源更稳。**SAM3/DINO 预训练泛化强 → 跨省更优**,是论文卖点。
- 漏检集中在**小地块**(route-c 完全漏检 14.8% 个数 / 7.2% 面积),正是 1m 该补的。
- 历史:DINOv2(v3–v12 patch 级撞 ~50–60% 顶;v39 DINOv2+UNet 密集分割,**一直输给普通 UNet**——但那都是 10m 任务,不代表 1m 地块分割也输);SAM3 之前只"评估过当分类器低 ROI",**从没真拿它做实例分割**(它的强项)。

## 数据 / 服务器(见记忆 server-network-topology / gpu-server-proxy)
- **GPU 服务器** `ps@10.147.19.250`(4×RTX4090);**store** `ps@10.147.19.174`(451TB,`/mnt/sdb/shared/zf`)。`.250/.149/.174` 光纤直连(~100MB/s),**跨服务器用 ssh 直推,别绕 Mac**。
- **Mac↔服务器走 ZeroTier,总带宽封顶 ~8MB/s**(不是单流限制);服务器联网慢(proxy ~4.5KB/s,基本没法重下)。大数据先在 Mac 端压再传。
- 路径:1m 融合数据 `/mnt/sda/zf/landform/data/c_1m`(大盘);10m 数据 `/home/ps/landform/data/{v19_s2_raw,v33_ndvi_multitemporal,v11_dltb}`;结果 `/mnt/sda/zf/landform/results`。
- **权重**:SAM3 在 **Mac `~/D/sam3/`**(+ HF `models--facebook--sam3/sam3.1`);DINOv2-large 在 Mac `~/D/dinov2_weights/` 和 .250 `/home/ps/landform/dinov2/dinov2-large`;SAM1 `~/D/test/sam_vit_b_01ec64.pth`。**SAM3 还没在 .250 上**(需传权重+推理代码+配环境)。

## 构建 / 运行
- **Rust/CLI**:`cd src-tauri && cargo build`(`cargo test` 全过)。CLI:`target/debug/imagery-downloader batch --regions <json> --out <dir> --source esri|google|auto --zoom 17 --compress jpeg|deflate|none --quality 95`。
  - 输出默认 **单文件 YCbCr JPEG-GeoTIFF**(~10×,地理信息完整;提交 0d75f36)。`cog.rs` 手写了 TIFF-JPEG(`tiff` crate 不支持 JPEG)。
- **sidecar**:在 .250 跑,python = `~/miniconda3/bin/python`(有 torch/smp/geopandas/rasterio)。并行用 ProcessPool / 多 GPU dispatcher。
- **GPU 充分利用**:用户要求训练任务尽量并行铺满 4 张卡(poll 式 dispatcher,launch 前查 GPU 真空闲,别硬编码 GPU/别按时间假设)。

## 工程坑(踩过的)
- macOS `rsync` 不认 `--info=progress2`;`cat` 不认 `-A`。
- ssh 里内联 `nohup ... &` 输出会截断 → 复位/启动用脚本 + `setsid` + `</dev/null`。
- ssh heredoc 里 f-string 转义引号会炸 → 用 `%` 格式化。
- stage-1 推理是 **IO 密集 + GPU 轻载**,别按 GPU% 判断进度;`torch.load(..., weights_only=True)`。

## 关键文件
- 研究台账 `sidecar/EXPERIMENTS.md`;经验总结 `sidecar/1M_FUSION_LESSONS.md`;1m 路线计划 `sidecar/ROUTE_C_PLAN.md`。
- 主要脚本:`fuse_1m.py` · `train_c_stage1.py`(`--sources`)· `stage1_infer.py` · `train_c_stage2.py` · `train_route_a.py` ·
  `ens_eval.py` · `polygon_miss.py`(对象级漏检)· `changzhi_*.py`(跨省)· `run_extra_dispatch.py`(GPU 调度)· `recompress_jpg.py`。
- 拆分:`v40_5k.json`(5k 训练子集,测试集与 v40_xcounty 同口径 120 cell/12 县)、`v40_xcounty_regions.json`(20k)。

## 用户偏好(见记忆 parallel-by-default)
中文沟通;研究要诚实(标注口径、不夸大);GPU/机器能并行就并行;论文要 benchmark vs FTW/Hou et al./DeepLabv3+/SegFormer/DINOv2。
