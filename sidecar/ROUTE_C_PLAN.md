# 1m-fusion 路线 (用户决策 2026-05-30)

破 10m 天花板(跨县 F1 0.853 / acc 0.906)的方向 = 把 1m 高分接进来。三条候选,**用户选 (c)**,(a)(b) 记录待测。

## 背景 / 已知
- 当前 8 模型集成(纯 10m 哨兵+多年 NDVI)跨县 F1=0.853、acc=0.906、IoU=0.73;跨省(长治)F1 0.56–0.72。
- 朴素 1m 融合 POC(把 1m 降到 5m/2.5m 当附加通道拼上去)**无增益**:12ch vs 9ch = −0.006(5m)/ −0.007(2.5m)。
  - 失败原因:① 降采样把 1m 价值砍了;② 1m 当附加而非主输入;③ POC 仅 800 cell(欠功率);④ Esri 单源时相不对齐。
- ⇒ 正确做法:**1m 原生分辨率 + 1m 做主 + 全量 + 多源**。

## 三条路线
- **(a) 单网络**:1m 原生网格,S2 上采样拼入,一个 UNet。最直接。**记录,待测。**
- **(b) 双支路**:1m 高分支 + 10m 光谱支,后融合(FiLM/concat)。**记录,待测。**
- **(c) 两阶段 ✅ 用户选(最优潜力)**:
  - **Stage-1(1m,6 通道 = Esri RGB + Google RGB,多源)**:1m 原生网格做分割,得到亚米级 sharp 边界(+边界损失锐化)。多源 = 两次独立成像,真边界两源都在(稳)、伪影/时相差单源出现(滤掉)。
  - **Stage-2(判类)**:stage-1 的 1m 边界/概率 + 10m S2 RGBNIR + 多年 NDVI → 用光谱时序纠正 RGB 看不出的类(绿草≠耕地)→ 最终耕地图(1m 边界,光谱定类)。

## 实现脚本(全部已写)
- 下载:app CLI `imagery-downloader batch`(全量 5120 cell,esri+google 双源,断点续传)。
- Stage-1 数据:`fuse_1m.py`(每 cell → 6ch@1m + DLTB@1m label,ProcessPool by-county)。
- Stage-1 训练:`train_c_stage1.py`(UNet 6ch,512px@1m,CE + boundary loss)。
- Stage-1 推理:`stage1_infer.py`(滑窗 512 推理 → 1m 耕地概率,area-pool 降采样到每 cell 的 10m 网格 → 2 通道:均值概率 + 边界密度,存 c_stage1_feat/)。
- Stage-2:`train_c_stage2.py`(9ch 光谱 + 2ch stage1 = 11ch,v36 配方;`--no-1m` 把 2 通道置零做消融)。
- 编排:`run_route_c_full.sh`(4-GPU 并行,见下)。

## 执行记录(2026-05-30)
**传输优化(关键):** 1m 影像 164GB 在 Mac;Mac↔服务器走 ZeroTier **总带宽封顶 ~8MB/s**(光纤只连服务器,Mac 不在网内),服务器联网 4.5KB/s 无法重下 → 数据必过慢链路。解法:Mac 端把 RGBA-tif **重压成 RGB JPEG-GeoTIFF(q92,10.3×)** → 164GB→15.6GB,地理信息保留,fuse 零改动(只读 RGB 三波段)。ship 8h→~33min。脚本 `~/recompress_jpg.py`。

**干净消融设计:** v40_5k 与 v40_xcounty **测试集完全相同**(同 120 cell / 同 12 留出县),区别仅训练量(5k vs 20k)。
- baseline = `train_c_stage2 --no-1m`(11ch 架构,2 个 1m 通道置零),5k 训练。
- route-c = `train_c_stage2`(真 1m 通道),5k 训练,**唯一差异 = 1m**。
- **1m 净贡献 = route-c − baseline**(同种子/同架构/同数据);再看能否逼近/超 0.853(20k 集成参考)。

**GPU 调度(run_route_c_full.sh):** GPU0 stage-1 ‖ GPU1 baseline(并行);stage-1 完→GPU0 推理→GPU2 route-c。预计 fuse 后 ~1.5h 出对照。

## 评估口径(不变,诚实)
甘肃跨县 12 留出县 + 长治跨省;报 F1 / IoU / accuracy。最终用 D4 TTA + 留县阈值 CV(诚实可迁移数)。对照基线 = 纯 10m 集成 0.853 / acc 0.906。
