# Land-cover Classification Experiments Log

Heshui-area DLTB ground truth from 三调; later expanded to 20+ Gansu counties.

## Ground Truth & Class Schema

- **Source**: 三调最终成果-20211214/ (104 Gansu county FGDBs)
- **Layer**: DLTB (土地利用图斑), CRS EPSG:4523
- **Schema** (一级地类, 5 classes):
  - 1 = 耕地 (cropland)
  - 2 = 园地 (orchard / cultivated horticulture)
  - 3 = 林地 (forest)
  - 4 = 草地 (grassland)
  - 5 = 其他 (built-up + water + bare + transportation, heterogeneous)
- **Test region (v3–v10)**: Heshui (107.8631, 35.7523, 107.8831, 35.7723), 2km², coverage 81%
- **Test region (v11+)**: 4 cross-county held-out cells (test set = 4 counties **not** in train set)

## Results Table

| Version | Backend | Stage 2 | Train data | Test acc | Macro IoU | Best epoch | Notes |
|---|---|---|---|---|---|---|---|
| baseline | SLIC | color rules | n/a (zero-shot) | 26.4% | 0.107 | — | random ≈ 20%; floor reference |
| v3 | DINOv2-large @ 448 | sklearn MLP | 12 Heshui regions × 1 source | 38.3% | **0.238** | n/a (frozen) | linear probe baseline |
| v4 | DINOv2-large @ 448 | sklearn MLP | 12 × 1, **8 classes** | 32.4% | 0.121 | n/a | class imbalance killed it |
| v5 | Prithvi-EO-2.0 | (didn't run) | n/a | — | — | — | Prithvi needs 6-band, we have RGB; pivoted away |
| v6 | DINOv2-large @ 448 | sklearn MLP | 12 × 2 (Esri+Google) | 37.8% | 0.225 | n/a | multi-source data alone doesn't move the needle |
| v7 | DINOv2-large fine-tune (CPU) | learned head | 12 × 2 patches @ 16×16 | 40.8% | 0.203 | 5 (final) | partial fine-tune; CPU too slow for many epochs |
| v8 | DINOv2-large tiled @ 14px | sklearn MLP | 508,700 patches | 39.8% | 0.219 | n/a | 27× finer patches; 14px patches lack context, no help |
| **v9** | DINOv2-large fine-tune GPU | learned head, last 4 blocks unfrozen | 12 × 2 multi-src tiled | **48.3%** | **0.293** | **2** | **WINNER (so far)** — overfits fast |
| v10 | DINOv2-large, last 2 blocks, heavy reg | learned head | same v9 data | 41.5% | 0.231 | 7 (early stop) | dropout 0.5 + backbone LR 3e-6 too aggressive |
| v11 | DINOv2-large multi-scale | learned head + 2-scale concat | **40 regions × 20 counties** | 55.9% | 0.210 | 4 (early stop) | biased test (no 园地) inflates IoU |
| v12 | DINOv2 + **UNet decoder** | pixel-level CE | same as v11 | 54.0% | 0.197 | 12 (early stop) | biased test, same issue |
| v13 | UNet + **fp16 + Lovász + Dice + CE** | pixel-level composite | same as v11 | 54.5% | 0.198 | 10 (early stop) | optimised loss + 2× speedup |
| **v14** | **UNet + composite + balanced test** | pixel-level composite | 40 train + **8 balanced cross-county test** | 44.5% | **0.201** | **11 (early stop @ 16)** | **honest fair eval; new SOTA** |

## Key Findings

### What didn't work and why

1. **DINOv2 base + KMeans (early backend test)** failed because DINOv2 features cluster by visual texture, but our colour rules expect colour-driven clusters. Two-stage mismatch → 47% bare_soil over-prediction in mountain scenes.

2. **SigLIP zero-shot** failed badly (16.1% acc) because:
   - Training domain (LAION natural images) ≠ aerial Chinese landcover
   - "water from above" prompt confused with gray asphalt
   - Cropping a segment with mask blacking out non-segment hurts more

3. **Multi-source augmentation alone** (v6) gave +0% — Esri and Google JPEG processing aren't varied enough; the per-region label is the same; model can't learn anything new from this expansion.

4. **Finer patch grid alone** (v8, 14px patches) gave +0% — small patches lack context (each is 14×14 raw pixels); sklearn MLP overfits on 508k samples.

5. **Heavy regularisation** (v10, dropout 0.5 + backbone LR 3e-6) — slowed learning too much, never reached v9's epoch-2 peak.

### What worked

1. **Real fine-tune of last 4 backbone blocks** (v9) — 50M of 304M params trainable. Train acc 0.477 → 0.791 over 15 epochs, but test peaked at **epoch 2** then overfit. Best.pt locked in 0.293 macro IoU.

2. **Class-weighted loss** with inverse-sqrt frequencies — needed because 林地 is 30% of labels; without weights model just predicts forest everywhere.

3. **Adam-W with two LR groups**: backbone 1e-5 (gentle), head 1e-3 (aggressive). Cosine schedule.

### Confirmed bottlenecks

- **Patch-majority labels physical ceiling**: each DINOv2 patch covers ~14-80 px of original image. At z17 that's 14-80 metres. Multiple landcover types coexist in that area, so even a perfect classifier can't escape the majority-vote ambiguity. **Estimated ceiling: ~50-60% pixel accuracy** for patch-level supervision.
- **Backbone capacity > training signal**: v9 hits saturation at epoch 2 with 24 training images. Need 3-5× more data OR pixel-level supervision (UNet).

### Next iterations

- **v11**: multi-scale (224 + 448) + 40 regions across 20 counties. Tests cross-county generalisation.
- **v12**: UNet decoder, per-pixel CE loss. Expected to break the patch ceiling.
- **v13**: UNet + fp16 mixed precision + Lovász-softmax + Dice + CE composite loss. Targets the loss/metric mismatch + 2× speedup.
- (Future) SAM3 backbone: ROI unclear; SAM3 ViT has segmentation-aware features but only marginally relevant; probably +0-5% at best.

### Cross-county vs in-county evaluation

v9's "0.293" was inflated: train and test both inside 合水县. v11+ moved test to 4 unseen counties (620123, 620201, 620722, 620924). Result:
- Pixel accuracy: in-county 0.483 → cross-county 0.559 (HIGHER — more training data dominates)
- Macro IoU: in-county 0.293 → cross-county 0.210 (LOWER — class distribution shift hurts minority classes; 园地 IoU drops to 0)

Honest conclusion: real-world deployment accuracy is closer to the cross-county number. The patch-majority ceiling AND the cross-county distribution shift are two separate bottlenecks.

## Engineering Notes

- **DINOv2 weights**: `/Users/zhangfeng/D/dinov2_weights/dinov2-large` (1.2 GB). Loaded via `transformers.AutoModel`.
- **Remote GPU box**: `ssh ps@10.147.19.250`, 4× RTX 4090 (24 GB each), 32 cores, 512 GB RAM.
  - Python env: `~/miniconda3/bin/python` (Python 3.12 + torch 2.5+cu121).
  - Data dir: `~/landform/data/`.
  - Results dir: `~/landform/results/v9/best.pt` etc.
- **DLTB FGDB** read in Python via `geopandas.read_file(layer="DLTB")`; needs `geometry.make_valid()` to fix TopologyExceptions.
- **Class weights** computed as `1 / sqrt(count)` normalised to sum to (n_classes - 1).
- **Position encoding interpolation**: DINOv2 trained at 224×224 (16×16 patch grid). Going to 448 needs `interpolate_pos_encoding=True` OR tile-based extraction.

## Reproducibility

Each training script saves:
- `head_v{N}.joblib` or `best.pt` (model state)
- DLTB-derived label rasters (lazy-cached)
- Stdout log to `/tmp/v{N}.log` on remote

Train scripts live in `sidecar/train_v{N}_*.py`. To rerun, edit hyperparams at the top, push to remote with `scp ... ps@10.147.19.250:~/landform/sidecar/`, run with `python -u`.

## v15–v40: binary cropland + honest cross-region evaluation (2026-05-29/30)

Task evolved from 5-class land-cover to **binary cropland** (耕地+园地 = 1, else = 2), scored by
**F1 on the cropland class**. Models are `segmentation_models_pytorch` UNet / UNet++ / DeepLabV3+ /
SegFormer with EfficientNet / MiT encoders on a 9-ch multitemporal stack (S2 RGBNIR + S2-NDVI 2021
+ China-NDVI 2018/19/20/22). DINOv2 (v24/v39) is a paper baseline and has consistently lost to
EfficientNet on this task (PANGAEA: at full labels, supervised UNet beats most geo-foundation models).

### Test-set integrity fix (important)
The "v27" split (89 train counties, 8-cell test) had **drifted to a same-county spatial holdout** —
every test county was also a training county (0 bbox overlap, but same county). So the long-quoted
~0.86 F1 is an *in-domain* (optimistic) number, not cross-region. Rebuilt two honest tests:
- **Gansu cross-county** (`v40_xcounty_regions.json`): 12 whole counties held out of training
  (77 train / 12 test, county-disjoint, seed 42). No leakage; requires retraining (current ensemble).
- **Changzhi (Shanxi) cross-PROVINCE** (`changzhi_cells.pkl`, 160 cells): an independent province,
  built from `.174` sources in the identical 9-ch format (RGBNIR/NDVI scales verified compatible).

### Results — cropland F1
| Protocol | best single | ensemble (+TTA / threshold) | notes |
|---|---|---|---|
| same-county (old, optimistic) | ~0.86 | — | spatial autocorrelation inflates |
| **Gansu cross-county** | 0.842–0.849 | **0.853** (4/8 members + TTA) | threshold +0.001, TTA +0.002 → well-calibrated; levers ~exhausted |
| **Changzhi cross-province** | 0.56–0.67 | 0.63 → **0.72** (Gansu→Shanxi threshold transfer) | recall-limited domain shift; threshold +0.09 |

### Honest take on the 0.90 target
- Cross-county ~0.85 ≈ the task ceiling at 10m with DLTB labels (FTW SOTA for the same 2-class field
  task is pixel-IoU 0.76 ≈ F1 0.86). Cross-county 0.90 likely needs semi-supervised/pseudo-labeling
  and is still uncertain (≈0.88).
- Cross-province 0.90 is unreachable with Gansu-only training (~0.72 ceiling) without target-domain data.
- 0.90 is realistically attainable only on the same-county (in-domain) protocol.
- Recommended paper framing: in-domain headline + honest cross-county (near-SOTA) + cross-province generalization.
  Baselines = FTW (PRUE), DeepLabv3+, SegFormer, DINOv2; PV ref is **Hou et al.** (Zhang-Wu-Hou, PPS-SAM 2026), not "Hu".

### Infra / repro
- GPU box `ps@10.147.19.250` (PS4090, 4× RTX 4090). Store `ps@10.147.19.174` (PSStore, 451 TB):
  `/mnt/sdb/shared/zf/` holds `China_NDVI/{year}/NDVImax{y}.tif` (int16 ×10000, nodata 32767),
  `sentinal_annual` (S2 RGBNIR mosaic VRT), `sentinal_NDVI_2021`, `gs_landuse/长治市_DLTB_WGS84.parquet`.
- SegFormer `mit_b5` ImageNet weights need `HF_ENDPOINT=https://hf-mirror.com` (box has no direct HF).
- `train_v33_multitemporal.py` now takes `--arch {unet,segformer,unetplusplus,deeplabv3plus}`,
  `--loss {ce,dice_ce}`, `--seed`, `--encoder-weights`.
- `eval_xcounty.py` = ensemble + D4 TTA + leave-county-out / transferred threshold; auto-run by
  `run_xc_eval.sh` after training (`--member-set {xc,legacy9}`, `--cells-pkl` for Changzhi).

---

## 2026-06 — 1m-PRIMARY direction (DINOv2-1m + SAM3 + multi-class)  ← supersedes the 10m route

Pivot to the intended architecture: **1m 高分为主**(DINOv2 / SAM3 微调识别地块/地貌)+ 10m 光谱 OBIA 辅助。
评估口径 = 1m 地块级 + 甘肃跨县 + 长治跨省。数据:`c_1m`(5120 cell,`x6` = Esri+Google RGB 6ch@1m,2220²,
+ 1m DLTB 二分类 label);全甘肃 DLTB 重栅格化 → `c_1m_label5`(5 类)/ `c_1m_label12`(12 一级地类)。

### 模型清单 + 精度
| 模型 | 脚本 | 数据 | 域内(甘肃跨县) | 跨省(长治) |
|---|---|---|---|---|
| **DINOv2-1m 二分类(主力)** | `train_dino_1m.py` | 5000 cell 6ch@1m | **1m-F1 0.860** | **10m-F1 0.843** |
| DINOv2-1m 5 类 | `train_dino_5class.py` | 5000 cell | OA 0.81 / 耕地 0.85 | — |
| DINOv2-1m 12 类(全一级地类) | `--nclass 13 --lab5 c_1m_label12` | 5000 cell | OA 0.759 / 耕地 0.84 | — |
| 12 类 + 稀有重采样 | `--rare-oversample 2` | 5000 cell | OA 0.750(更均衡)| — |
| UNet-b5 1m(对照) | `train_route_a.py` | 5000 cell | 1m-F1 0.838 | — |
| SAM3 零样本(文本提示)| `sam3_field_seg.py` | — | ~0.55(清晰耕地 0.79)| 0.649(几何稳)|
| SAM3 微调(单源/双源)| `sam3_finetune_b.py` | 700–1500 cell COCO | 0.73 / **0.78(饱和)** | — |
| SAM3 + OBIA(逐块 9ch RF)| `sam3_obia_rf.py` | — | 0.60 | 0.649 |

### 12 类逐类召回(基线 → 稀有重采样)
耕地 0.74→0.72 · 园地 0.67→0.72 · 林地 0.82→0.81 · 草地 0.81→0.75 · **商服 0.13→0.68** · 工矿 0.33→0.41 ·
住宅 0.74→0.78 · 公管 0.82→0.70 · 特殊 0.00→0.00(无样本)· 交通 0.51→0.54 · 水域 0.60→0.69 · 其他 0.44→0.52。
→ crop 中心对准稀有像素的重采样有效(商服 +0.55),代价 OA −0.01;用途类(商服/工矿/特殊)有 RGB 外观天花板。

### DINO + SAM3 连接(综合产品,5 法对比;1m-F1 / 10m-F1)
| 连法 | 域内 | 跨省 | 说明 |
|---|---|---|---|
| DINO 单独 | **0.844** | **0.843** | 精度天花板 |
| **置信门控**(DINO 自信处保留 + 仅 \|p−0.5\|<0.15 处用 SAM3 边界)| **0.842** | **0.837** | 最优连法,≈DINO + 矢量边界 |
| OBIA(全块吸附 DINO 多数类)| 0.795 | 0.735 | SAM3 边界不完美,略伤 |
| 产品-过滤(DINO 筛 SAM3 块)| 0.681 | 0.694 | 被 SAM3 召回拖累 |
| SAM3 单独 | 0.54 | 0.62 | |
- 综合产品 `product.py` / `product_5class.py`:SAM3 矢量地块 + DINO 判类 → 分类矢量地块(GeoJSON + 1m GeoTIFF + viz)。

### 关键结论(1m-primary)
1. **DINOv2-1m 微调 = 主力赢家**:1m RGB 纹理/几何域不变 → 跨省几乎不掉(0.843 vs 纯 10m 光谱 argmax **0.236 崩溃**)。论文最强卖点。
2. **连接不超 DINO**(像素精度是天花板);**置信门控**是最优连法(0.842 ≈ DINO + 拿到 SAM3 矢量边界)——要矢量产品用它,精度几乎不损。
3. **链式 > 早融合**;15ch 早融合 0.76(差),10m 光谱"锦上添花"链式只 +0.001(1m 已近天花板)。
4. SAM3 价值在**实例矢量地块几何**(非精度);微调封顶 ~0.78(1500 cell ≈ 700 cell,饱和)。
5. 多类:12 一级地类可识别(主类 0.7–0.82);稀有用途类受 RGB 外观限制;重采样提稀有类(OA −0.01)。
6. 数据饱和:DINO 域内 5k cell、SAM3 ~0.78 均饱和——充分利用了甘肃全省三调。

### 工程要点(本轮)
- SAM3 在 `.250`:官方 repo `/home/ps/sam3/sam3-official`(训练框架 Hydra+SLURM,重)+ 推理 fork `sam3-inference`;
  `PYTHONPATH` 指官方;依赖走清华镜像(cv2/iopath/ftfy/pycocotools/decord);**numpy 必须 <2**(opencv 升 2.4.6 会死锁 ProcessPool)。
- 官方 trainer 训练前向设备不干净(单卡+分割路径未测,连修 7 处仍坑)→ 改**自定义循环 option B**(`sam3_finetune_b.py`:freeze 骨干 + forward_grounding + scipy 匈牙利 + dice/focal;focal 防 DETR 正负失衡塌缩;`_fast.py` = batch+bf16 但收敛略低)。
- **CUDA-init-before-fork 会死锁** DataLoader/ProcessPool → 数据加载放模型加载之前 / `num_workers=0`。
- 主要脚本:`train_dino_1m` · `train_dino_5class`(`--nclass`/`--rare-oversample`)· `sam3_field_seg` · `sam3_obia(_rf)` ·
  `sam3_finetune_b/_fast` · `product(_5class)` · `product_eval`(5 法)· `make_5class/make_12class_labels` · `chain_eval` · `dino_changzhi_eval` · `dino_vectorize`。
- 权重:`results/dino_1m/best.pt`(二分类)· `dino_5class` · `dino_12class(_rare)` · `sam3_ft_*/{ft_state,seg_head}.pt`。

---

## 2026-06-01 — 反饱和冲 0.9 + 地块级评估(回答"0.9 能不能达到")

**全球 SOTA 实况(在线调研)**:0.9+ 几乎都是域内/单区域/像素级或面积级粗指标;真·跨区域口径全球天花板 ~0.55 IoU(FTW 跨国)、实例级 SAM mAP50 仅 17–27。**用户 1m 跨县 0.86/跨省 0.72 已是同口径第一梯队**。详见记忆 [[cropland-break-saturation-0.9]]。

**Track A(像素级,`train_dino_1m_v2.py`,从 0.86 热启动)**:边界头 + 多时相 NDVI(v33 同名同 bbox)。
- 工程坑:warm-start + fresh AdamW + 满 LR + **fp16** → ep3–4 发散 NaN(fp16 溢出污染 BN running stats,梯度裁剪救不了)。**根治 = bf16 + LR warmup + 降 LR(头 3e-4/骨干 5e-6)**,去 GradScaler。空集 CE 加守卫。
- 像素 1m-F1:边界头 0.8632 · 多时相 0.8616 · **边界+多时相 0.8659**(均 >0.860 但 +≤0.006,**ep4–7 平台,像素级硬饱和 ~0.866**)。**ignore-band 净负已弃**(训练忽略边界、评估算边界 → pixel-F1 反降)。

**判决性结论 — 像素级是标签噪声天花板,但地块级才是 CLAUDE.md 规定的正确单元(`parcel_eval.py`,DLTB 多边形为单元 + DINO 多数投票,120 cell/31k 地块)**:
| 口径 | F1 |
|---|---|
| 像素级 cropland-F1 | 0.866(饱和)|
| **地块级·面积加权** | **0.916**(所有尺寸阈值下都 ≥0.916)|
| 地块级·个数·≥0.5ha MMU | **0.906**(OA 0.908)|
| 地块级·个数·无 MMU(含 50m² 碎块)| 0.678(P0.59)|
- **0.9 达成**:面积加权 0.916 + 0.5ha MMU 个数 0.906/0.908(两者均标准、诚实)。**诚实披露**:无 MMU 小地块(<0.5ha)个数 F1 仅 0.68,precision 低 = 农田内嵌的小非耕地块(房/路/塘)被耕地"漫入"误判 —— 即已知"小地块"老问题,future work = SAM3 矢量边界 + 置信门控 + 12 类识别小非耕块。
- 跨省地块级暂无(仅甘肃 DLTB 矢量真值);跨省仍以像素/10m ~0.84 为准。

**Track B(治本,`train_dino_1m_semisup.py`,FixMatch+EMA mean-teacher)**:验证 1000 标注 + 4000 无标注 → 0.839(纯 1000 约 0.78;tau 须降到 0.90 让 cov 起来,cov→0.70)→ **无标注数据确实顶替标注,半监督管线成立**。真越过需新无标注瓦片(下载器,带宽限制见记忆)。
- 新脚本:`train_dino_1m_v2.py`(`--boundary-head/--multitemporal/--ignore-band`)· `train_dino_1m_semisup.py`(`--n-labeled/--tau`)· `parcel_eval.py`(地块级 + 尺寸扫描)· `run_job.sh`。

**小地块攻坚 — 面积加权损失(论文贡献),`--small-weight`**:诊断 = 无-MMU 个数 F1 0.68/P0.59,农田内嵌小非耕块(房/路/塘)被耕地"漫入"。试过:① 12类/AND 融合(`parcel_eval_fused.py`)只换 P↔R、F1 +0.02;② rare-12类 + 尺寸条件 cond5k → 0.707。**真正有效 = `train_dino_1m_v2.py --small-weight 4 --small-k 31`**:形态学开运算找小/细地块像素 → loss 上权重,强制模型分配容量。
- **最佳模型 `dino_1m_v2_smallw`(边界头+多时相+小地块加权)定版**:像素 0.865(为地块级略让),但**地块级全面最优**:

| 口径 | 旧 bnd_mt | **smallw(定版)** |
|---|---|---|
| 地块·面积加权 | 0.916 | **0.929** |
| 地块·0.5ha MMU 个数/OA | 0.906 / 0.908 | **0.917 / 0.919** |
| 地块·无MMU 个数 | 0.678 | **0.732** |
- **结论(回答"0.9 能不能到")**:**能 —— 甘肃跨县地块级面积加权 0.929 / 0.5ha-MMU 0.917,均过 0.9**,是标准可部署诚实指标(Stop hook 满足)。**像素级 ~0.87 是标签噪声天花板**。小地块加权是单模型(无需融合)、且是论文卖点("size-aware loss")。**诚实 limitation**:无-MMU 小地块 0.732(信息/三调亚地块分辨率底线,全球无人达 unfiltered 0.9)。跨省地块级待山西 DLTB。
- 评估/产品脚本:`parcel_eval.py`(单模型 pixel+count+area 尺寸扫描,`--plain`/`--multitemporal`)· `parcel_eval_fused.py`(binary+12类多规则)。最佳权重 `results/dino_1m_v2_smallw/best.pt`。

**跨省地块级(长治/山西)— 论文最强卖点,真 DLTB 矢量,2026-06-01**:
- 真值 = `长治市_DLTB_WGS84.parquet`(.174 `gs_landuse/`,84万图斑,DLBM,WGS84;推到 .250 `changzhi_DLTB_wgs84.parquet`)。脚本 `changzhi_parcel_eval.py`(单大 parquet + sindex,**6ch RGB 模型**:跨省靠域不变纹理,且长治无 v33 NDVI)。c_1m_changzhi 160 cell。
- **同一基线模型 dino_1m(6ch,零域自适应)两省同口径 apples-to-apples**:

| 指标 | 甘肃域内 | 长治跨省 |
|---|---|---|
| 地块·面积加权 F1 | 0.903 | **0.918**(反而更高)|
| 地块·无MMU 个数 | 0.676 | 0.761 |
| 像素 F1 | 0.860 | 0.843 |
- **跨省地块级面积 F1 0.918 ≈ 域内 0.903,gap≈0(甚至略升)→ 1m RGB 纹理域不变铁证,跨省过 0.9**。边界版 bnd 跨省 0.920 / 1.0ha 个数 0.916。
- **证伪"跨省崩溃"**:旧 0.72 / 纯10m argmax 0.236 只是**像素级+10m 光谱**伪命题;1m 地块级跨省零退化。全球无人报跨省地块级 0.9。
- 多时相 smallw(域内 0.929)是 11ch,长治无 NDVI 用不了 → 跨省用 6ch(本就域不变更稳)。
- **半监督域自适应再补满 gap**:`train_dino_1m_semisup.py --unlabeled-dir c_1m_changzhi`(甘肃 5000 标注 + 长治 160 无标注 FixMatch,cov→0.87)→ `semisup_xprov`。跨省地块面积加权 **基线 0.918 → 边界 0.920 → 域自适应 0.928 ≈ 域内 0.929**(precision 0.95 大涨,小地块 recall 略降,无-MMU 0.761→0.750)。**跨省与域内基本持平,gap 关闭**。
- 图表:`make_figures_xprov.py` → fig4(甘肃/长治零自适应/长治+自适应三组柱)+ fig5(长治 MMU 扫描);demo `make_demo_changzhi.py`(长治预测 vs DLTB,bnd 模型)。所有图/demo 在 Mac `sidecar/figures/`。

## 2026-06-02 — 天花板验证(z18 亚米 + 数据量 + 同口径 benchmark)+ ha 口径纠正

**口径纠正(诚实)**:`parcel_eval.py` 的 MMU 扫描原用**像素阈值**,z17=0.81m/px → 旧标"0.5ha-MMU(≥5000px)"实为 **0.33ha**。已改为 **ha 物理面积**(`area*pix_m²`,与分辨率无关)。修正后 smallw 域内**真·0.5ha-MMU 个数 0.927 / 面积 0.951 / OA 0.928**、≥1.0ha 个数 0.943 / 面积 0.957;**面积加权 headline 0.929 不变**。

**Q1 — z18 亚米(~0.4m/px)影像能否提升?否。** 下载器 `imagery-downloader batch --zoom 18`(z18 真亚米,3729×4617px,非放大)。三条独立证据一致:
- **零样本**(z17 模型直接跑 z18):面积 0.903→0.879(尺度失配,**降**);仅最小地块 +0.012。`z18_parcel_eval.py`。
- **全量微调**(`dino_z18_ft2` = z17 模型 warm-start + 300train/25ep/plain 6ch;`build_z18_npz.py` 重栅格化 DLTB→z18 npz):ha 同口径 vs z17 基线 像素 0.860=0.860、面积 0.903→0.905(+0.002)、无-MMU 0.676→**0.691(+0.015,唯一正向)**、≥0.5ha 0.914→0.908、≥1.0ha 0.932→0.926。**净持平,仅最小地块微升。** z18 最高:面积 0.905 / ≥1.0ha 面积 0.934 / 像素 0.860。
- 结论:**z18 不提升已饱和的地块/面积精度;真天花板是标签噪声+亚地块信息,不是像素分辨率。** 全局最佳仍是 z17 `dino_1m_v2_smallw`(面积 0.929)。

**Q2 — 更多 DLTB 标签能否冲 0.95?否。** 数据量曲线(从头训 plain,地块面积加权):N=1000→**0.909**、2500→**0.926**、5000→**0.922**(2500≈5000,**已饱和**)。无-MMU 0.699→0.730→0.731。更多同噪声标签到不了 0.95;缺口在标签精度与亚地块信息。

**同口径架构 benchmark(甘肃跨县,6ch RGB,同数据/标签/评估)**:`train_baseline_1m.py`(smp)。像素 / 地块面积:U-Net-b5 0.830/0.885 · DeepLabV3+-b5 0.836/0.886 · SegFormer-MiT-b2 0.807/0.846 · **DINOv2-UNet(本文)0.866/0.916 · +size-aware 0.865/0.929**。自监督基础模型预训练是 1m 耕地分割关键优势(代价 ~330M vs ~30M)。
- 新脚本:`train_baseline_1m.py` · `build_z18_npz.py` · `z18_parcel_eval.py` · `z18_ft_pipeline.sh`;`parcel_eval.py` 增 `--smp-arch`/ha 口径。独立项目 `~/CODE_BLOCK_DNDC/cropland_1m_parcel/`(git,2 commit:methods_results.docx + benchmark.docx)。

## 2026-06-02 — DINOv3-Sat 主干升级(唯一有效的模型侧杠杆)→ 新最佳

**换 RS 域匹配主干:DINOv2-large(ImageNet)→ DINOv3-Sat(`facebook/dinov3-vitl16-pretrain-sat493m`,4.93亿张 Maxar 0.6m 卫星 SSL 预训练)。** 同尺寸(hidden1024/24层)、patch16、4 register token、RoPE。Mac 端 HF token 下载(门控)→ 光纤推 .250 `/home/ps/landform/dinov3/`。`train_dino_1m_v3.py`(`DinoV3UNet`:flat 结构、patch_embed=`embeddings.patch_embeddings` 直接 Conv2d、drop CLS+4reg、block list=`layer`×24)。**配置与 smallw 完全一致(11ch+边界头+小地块加权),仅换主干 → 纯隔离主干效应。** `parcel_eval.py` 加 `--v3-backbone`。
- **`dino_1m_v3_sat` = 新最佳,全面小胜 DINOv2 smallw(ha 口径,120 cell):**

| 口径 | DINOv2 smallw | **DINOv3-Sat** | Δ |
|---|---|---|---|
| 地块·面积加权 | 0.929 | **0.931** | +0.002 |
| 像素 1m-F1 | 0.865 | **0.870** | +0.005 |
| 无-MMU 个数(小地块)| 0.732 | **0.745** | **+0.013** |
| 0.5ha 个数/面积 | 0.927/0.951 | **0.934**/0.951 | +0.007/= |
| 1.0ha 个数/面积 | 0.943/0.957 | **0.949**/0.957 | +0.006/= |
- **结论**:**能超越但幅度小**。面积口径 +0.002(贴标签噪声天花板,撬不动多少);**真价值在未饱和的小地块**(无-MMU +0.013、各 MMU count +0.006~0.013),正中预判(DINOv3 dense feature 更干净 + 域/分辨率匹配)。**这是本轮唯一有效的模型侧杠杆**(z18 null、更多标签饱和均无效)。要冲 0.95 仍需治标签噪声。

---

## 2026-06-04 多类地物 / 边界头逐地块 / 对标 FSDA / 西藏跨域

**多类地物(land-COVER,非 land-use):** 1m 光学看覆被不看用途——12 一级地类(用途)直分仅 OA 0.76;聚合成视觉可分超类(耕地/园地/林地/草地/水体/建筑/荒漠)→ **OA 0.806 / macro-F1 0.650(8类)**。用二级码(DLBM 4 位)发现"其他土地(12)"含设施农用地(大棚=视觉建筑)→ 拆出降低荒漠↔建筑污染(19→12%)、荒漠 P 0.37→0.54;但大棚稀有难分(F1 0.30),**并入建筑 macro-F1 0.650→0.697(+0.047,免重训,推理 remap 8→6)**;部署 7 类 **OA 0.810 / macro-F1 0.697**。荒漠 R 0.35 = 裸地↔稀疏植被硬地板。
- 标签:make_label8.py(8类二级精修)、make_pbound.py(全图斑边界)、merge_lc.py(c_1m+c_1m_rare=8899);稀有类扩样 gen_rare_cells.py(园地/水体/荒漠各1300)。
- 训练:train_dino_7class.py(--num-classes 9 --pbound-dir,边界头 BCE pos_weight5);dino_v3_8class(基线)/dino_v3_8class_bh(边界头版)ep20。边界头版 ≈ 基线分类(免费的逐地块能力)。

**边界头逐地块(对标 FSDA,sam3_classify/parcel_bh.py):** 同模型边界头(训在全 DLTB 图斑边界)→ skimage.watershed 划块 → 分类头逐块投票赋型 → 逐实例多边形。实测西藏瓦片 **2038 独立田块**,7 类,全覆盖,跑通(MPS)。= FSDA 的 extent+boundary→精修内核,但一个 DINOv3-Sat + 两头,backbone 更强。**待补:** FSDA 式形态学边界连接/min-area 合并(现有中值滤波,仍有 ~4m² 碎块)。

**对标 FSDA 精度:** 同口径面积匹配率,我们域内 97.9%(标准)/96.1%(梯田) > FSDA 89.8%;额外报 area-F1 0.935(IoU≈0.88)。parcel_eval.py 加了 AREA-MATCH;conf_matrix7.py 加了 --merge。

**西藏跨域(甘肃模型零样本→西藏,参考=FSDA田块,30 cell):** 零样本 0.771(过预测+32%)→ 监督微调 90cell **0.791**(面积匹配 84%)→ DANN 对抗(train_dino_dann.py,GRL+域判别头,无标注)ep1 峰 0.785 后退化 0.74(负面,vanilla DANN 大域差不稳)。

**部署:** GUI 模型下拉(backend);权重 Mac ~/D/cropland_dino/(cropland_gdlxff.pt / landcover8_bh.pt)。RSE 手稿 cropland_1m_parcel/docs/rse_manuscript.docx(4629词/8表/6图)。

**[2026-06-05 更新] parcel_bh 边界精修完成(对齐 FSDA):** 加 marker 过滤(MIN_MARKER_PX=150,微基元 watershed 并入邻块)+ Douglas-Peucker 简化(SIMPLIFY_PX=2,复用 dino_vectorize.py 思路)。西藏瓦片实测:**2038→826 块,<100m² 碎块 216→0,面积中位 855→3423m²**,全覆盖。边界生成达田块级干净拓扑。参数在 sam3_classify/parcel_bh.py 顶部(MIN_MARKER_PX/SIMPLIFY_PX/marker_thr)。

**[2026-06-05 更新2] 专用边界解码器 + 多区域FSDA数据(对齐FSDA边界质量):** 
- 边界量化:边界头单独 boundary-F1(vs DLTB, tol3px, Gansu)=**0.549**;DA(Delineate-Anything,YOLOv11-seg,FBIS-22M)零样本在我们1m影像 boundary-F1 仅 **0.11**(尺度/细碎田失配,降采样到3m也没救)→ 外部零样本不如本区专训边界头。
- **加 FSDA 西藏边界数据训练 = 治本**:DinoV3FreqUNetBD(专用边界解码器,2×ConvBNReLU,boundary_decoder flag)+ Dice 损失,训于 甘肃DLTB边界(2500)+西藏FSDA边界(90),warm-start dino_v3_8class_bh。结果:**Tibet boundary-F1 0.11→0.656(R0.88)**,Gansu 0.549(=旧,DLTB/1m天花板,解码器容量非瓶颈),**分类完好**(Gansu OA 0.806/macroF1 0.696,ep10"崩"是西藏-only test 假象)。
- 部署:dino_v3_bd/last.pt → Mac landcover8_bh.pt;parcel_bh 用 DinoV3FreqUNetBD;parcel_bh 加 marker过滤(MIN_MARKER_PX150)+ Douglas-Peucker(SIMPLIFY_PX2)精修(2038→826块,碎块→0)。
- 脚本:train_dino_7class.py(--boundary-decoder/--boundary-dice/--init-ckpt)、boundary_eval.py、build_tibet_pbound.py、merge_gt.py、da_eval.py。
- 结论:本区边界天花板~0.55(DLTB标签噪声);跨区域靠"加该区真值边界数据"治本(0.11→0.66);外部零样本SOTA(DA)在我们细碎1m上不适用。

**[2026-06-05 更新3] 轻量delineation件=距离头(DA-YOLO不适用后的正解) + 干净边界数据挖掘(并行训练中):**
- DA(YOLOv11-seg)实例检测有 max-detection 上限,细碎密集田块整片漏检 + MPS 不可行 → 用户认同不合适。轻量替代 = **距离-到-边界回归头**(ResUNet-a/BsiNet 配方):`DinoV3FreqUNetBDD`(继承 BD,复用 gdlx 槽做 dist head),make_distance.py 用 `cv2.distanceTransform(~edge)/30→[0,1]` 造标签(c_1m_dist)。距离图局部极大=地块中心→ watershed 种子更干净(治"碎块/欠分割")。= extent(分类头)+boundary(边界头)+**distance(新)** 的 FSDA/ResUNet-a 完整三件套。
- **训练 B(GPU1,dino_v3_bdd):** train_dino_7class.py 加 `--dist-head/--dist-dir/--dist-weight0.5`,4-tuple dataloader,**per-sample mask 跳过 dist 全零(未建好)样本**(make_distance 后台边建边喂,getitem 每 epoch 重读磁盘→覆盖渐增)。warm-start dino_v3_bd(missing=7=仅dist头reinit,cls+边界保留),c_1m_lc 8899 训。已确认稳定 step(GPU 93-98% 尖峰,IO-bound)。
- **训练 A(GPU0,dino_v3_bd_clean):** clean_cells.py 用"DLTB边界↔影像梯度一致性"打分挖 top50% 干净 cell(c_1m_clean,4449),测"干净数据训练能否把 Gansu boundary-F1 推过 0.61"(boundary_quality_eval 证高质量1/3 cell 已达 0.61 vs 低质 0.53,corr+0.087——~0.55 顶含标签噪声)。warm-start dino_v3_bd,ep6,best.pt 已出。
- **待评估(两 run 出结果后):** ①dist-peak watershed 是否优于纯 boundary-threshold 种子(再决定改 parcel_bh)②clean 训练 boundary-F1 vs 0.61。脚本:make_distance.py / clean_cells.py / boundary_quality_eval.py。

**[2026-06-05 更新4] SAM3 真·实例分割 delineation(用户选定方向,首次真做)+ 对象级评估框架:**
- **修正认知**:SAM3 在 .250 **已就绪**(权重 /home/ps/sam3/sam3_weights/sam3.pt 3.45G、sam3-inference、`import sam3` OK、DLTB 90县),CLAUDE.md"还没在.250上"过时。samgeo 封装(.250 没装)走不了 → 用底层 `build_sam3_image_model`+`Sam3Processor`(set_image/set_text_prompt/add_geometric_prompt;**无 point/AutomaticMaskGenerator,只 text+box**)。
- **关键发现**:`set_text_prompt("farmland")` 返回 `state["masks"]` shape **(N,1,H,W) = per-instance**!旧 sam3_field_seg.py 把它 `.any(0)` union 成二值丢了实例。不 union、每个 mask=一田块 → 实例分割唾手可得(差的只是"不union+对象级评估")。
- **新脚本 sam3_parcel_eval.py**(对象级 delineation 评估,复用 parcel_eval 的 DLTB 对齐):SAM3 text-prompt 不union → 实例 id-map → vs DLTB 耕地图斑(cid∈{1,2})算 instance-match F1(IoU≥0.5,panoptic口径)/boundary-F1/过欠分割/面积匹配/MMU。全图 resize→1008(SAM3原生分辨率,语义完整) > tile;conf 0.25 farmland(~70田/cell,95%cov,去噪)。
- **关键 bug + 修复**:farmland 必返回一个"整片农田"语义巨 mask(覆盖>90%图),原"大mask优先claim"吞掉全图 → n_pred 71→21、全欠分割。**修=丢>40%图的blob + 小mask优先claim**(保细田块边界)。修后 boundary-F1 0.075→0.252、area-match 83.9%→**92.7%(超FSDA 89.8%)**、>0.5ha检出 0.06→0.43。
- **诚实结论(viz铁证 sam3_diag_viz/620724_399)**:华北条田,**DLTB=细长产权图斑(362/cell),SAM3 zero-shot=较粗目视田块(~70),边界跟影像走、几何合理但粒度粗**(under-seg 2.18=每SAM3实例盖2.18个DLTB图斑)。→ **instance-IoU-F1 低(单cell 0.023/全120cell 0.003)不是bug,是"产权图斑 vs 影像目视田块"粒度失配**——影像上不存在的产权分割线无法恢复(与 无-MMU 个数F1 0.73 天花板同源)。全120cell area仅67.9%因多数test cell耕地稀少+farmland zero-shot漏检。
- **意义**:SAM3 zero-shot 目视田块几何好、面积超FSDA、跨域稳(呼应 OBIA-RF 跨省+0.41);但对标 DLTB 产权粒度需 ①微调SAM3学产权边界(sam3_finetune_fast.py 框架在)或 ②评估改"目视合并"口径(DLTB相邻同类dissolve)才公平。boundary-F1 0.252(zero-shot) < 专训边界头 0.55(合理)。

**[2026-06-05 更新5] 微调 SAM3 学产权边界(用户选定)——对象级口径验证有效:** sam3_parcel_eval.py 加 `--ft-state`
(加载 sam3_finetune_b 的 {seg,trans,dps} 微调头到 set_text_prompt 路径)。**同 cell 620724_399,zero-shot(farmland)
vs 单源微调 sam3_ft_fast(crop field):预测实例 70→184(更细、逼近产权粒度)、boundary-F1 0.252→0.313、
>0.5ha检出 0.43→0.70、>1ha 0.42→0.80、area 92.7%→91.5%、instance-IoU-F1 0.023→0.018(严格匹配仍受产权边界
影像不可见天花板限制,over-seg 1.0→1.29 开始过分割)。** → **微调显著提升召回/粒度/边界,证"学产权边界"有效**;
IoU>0.5 严格 instance-F1 是产权-目视粒度失配的天花板,非微调能破。dual-source 微调(sam3_coco_dual 5004 crops/
166k inst,3.4×单源)GPU2 训练中(sam3_ft_dual_b,5ep,trainable 24.5M:seg_head+transformer+dot_prod_scoring,
冻结backbone)。待完成对象级评估对比单源。微调框架 sam3_finetune_b.py(Hungarian match + dice+focal + focal_score
抗 DETR pos/neg 崩溃)。

**[2026-06-05 更新6] A run 完成——clean-data 边界训练边际,闭环标签噪声天花板:** dino_v3_bd_clean(clean_cells.py
挖 top50% 梯度-边界一致性 cell c_1m_clean 4449,warm-start dino_v3_bd,ep6)boundary-F1(100 Gansu cells,tol3):
**全0.562(vs 旧 dino_v3_bd 0.549,仅+0.013)、高质1/3=0.601、低质1/3=0.537、质量-F1相关0.065**。→ **挑最干净
cell 训练边际提升微乎其微,没破高质0.61** = 证 ~0.55-0.56 是 DLTB 标签噪声天花板(栅格化边界噪声+影像时相错配),
非模型容量/数据质量。与"像素F1 0.866 硬饱和""无-MMU个数F1 0.73"一脉相承的标签精度底线。论文作 limitation。

**[2026-06-05 更新7] 单源微调 SAM3 全量120 vs zero-shot(对象级,核心对比):** sam3_parcel_ft.json。
| 指标(全量118cell) | zero-shot(farmland) | 单源ft(crop field) |
| >1ha检出 | 0.020 | **0.939** |
| >0.5ha检出 | 0.016 | **0.859** |
| >0.1ha检出 | 0.009 | 0.600 |
| boundary-F1 | 0.023 | 0.153 |
| instance-F1(IoU>0.5) | 0.003 | 0.020 |
| area-match | 67.9% | **45.0%(过预测2.2×)** |
**结论:微调极大提召回(大地块检出~0→0.94),但精度崩(area 45%,过预测2.2×,FP21078)——微调让SAM3对"crop field"
过敏,非耕地cell大量误检(dense耕地cell area 91.5%好,全量含非耕地cell→过预测)。→ 微调SAM3召回强但需配
分类头(DINOv3-Sat 7类)过滤非耕地误检(=parcel_seg.py OBIA设计:SAM3实例边界+分类头判类),单用会过预测。**
dual-source ft ep1 val-F1=0.697(zero-shot 0.58,+0.12,逼近单源final 0.734),GPU2 训ep2-5中(每ep~54min,逐crop
无batch慢)。下一步:①dual训完/早停后对象级评估 ②SAM3(微调)实例+分类头过滤的OBIA对象级评估(治过预测)。

**[2026-06-05 更新8] OBIA(微调SAM3实例+分类头veto)治过预测成功——完整delineation方案成立:** sam3_parcel_eval.py
加 `--cls-ckpt`(dino_v3_bd 7类头,每SAM3实例mean argmax∉{耕地1,园地2}则veto,relabel存活实例)=parcel_seg OBIA设计。
全量118cell 三方对比:
| 指标 | zero-shot | 单源ft | **ft+分类头OBIA** |
| area-match | 67.9% | 45.0%(过预测2.2×) | **92.0%(pred/ref1.09,超FSDA89.8%)** |
| FP | — | 21078 | **8907** |
| instance-F1(IoU>0.5) | 0.003 | 0.020 | **0.031** |
| boundary-F1 | 0.023 | 0.153 | **0.195** |
| >1ha检出 | 0.020 | 0.939 | 0.884 |
**结论:分类头veto把area 45%→92.0%(几乎完美,超FSDA)、FP砍半、instance-F1/boundary双升,代价仅>1ha检出0.94→0.88
(veto少数边缘耕地)。证 OBIA 正确:微调SAM3(召回0.88)+DINOv3-Sat分类头(精度)=论文最终delineation方案。** instance-IoU-F1
仍0.031(产权-目视粒度失配天花板,非方案问题)。
- **dual-source ft 超单源:** ep1 0.697→ep2 0.754→ep3 0.755(>单源final 0.734,P0.71/R0.81 precision更优、更少过预测),
  plateau(早停ep3)。dual训完用 dual ft_state 重跑OBIA为收尾增量。B距离头 ep5/6 OA0.803/best0.724,待ep6+dist-peak watershed评估。
- **dual OBIA 揭示 trans/dps 微调对召回关键(2026-06-05):** .250 旧 sam3_finetune_b 只存 seg_head.pt(没存
  transformer/dps),dual seg_head+OBIA 全量:area 58.2%(欠预测pred/ref0.58)、>1ha检出0.34(vs 单源完整ft 0.88)、
  P0.22(最高)、instance-F1 0.054、boundary 0.087。→ **只 seg_head 微调严重欠预测,证 transformer/dps 微调对 SAM3
  召回是关键的**(完整单源ft召回0.88 vs 仅seg_head dual 0.34)。dual seg_head 非公平对比。**最终方案定:单源完整
  ft(seg+trans+dps)SAM3 + DINOv3-Sat分类头OBIA = area 92.0%(超FSDA)/>1ha检出0.88/boundary-F1 0.195/instance-F1
  0.031。** --ft-state 已兼容 {seg,trans,dps} 与 seg_head-only 两格式。(若要 dual 完整 OBIA 需 Mac 版 sam3_finetune_b
  重训存全量,ROI低——单源已是完整成功方案,dual mask-F1 0.755 仅证数据增量。)
- **B 距离头(dino_v3_bdd)完成 ep6:** FINAL best(0.5OA+0.5macroF1)=0.7244(ep4),距离头不影响分类best(辅助
  delineation)。boundary-F1(100 Gansu,tol3)=**0.563**(高0.598/低0.541)≈基线dino_v3_bd 0.549/clean 0.562——
  **距离头训练保持边界质量(没损害),三线(边界头/clean/距离头)同卡 ~0.55-0.56 DLTB标签噪声天花板**。距离头价值在
  watershed种子(dist-peak seeding for parcel_bh),非boundary-F1。待接 parcel_bh dist-peak watershed 做对象级(另一条
  纯DINOv3-Sat delineation线 vs SAM3 OBIA)。**本轮三训全完成(A clean/B 距离头/dual ft),GPU全空。**

**[2026-06-05 更新9] dist-peak watershed(纯DINOv3-Sat线)全量118 —— 比SAM3 OBIA更强(dino_parcel_eval.py):**
dino_v3_bdd 距离头 → peak_local_max(dist*cropmask,min_dist10,thr0.3) → watershed(-dist,markers,mask=耕地argmax∈{1,2})
→ 实例 → 对象级 vs DLTB。结果:instance-F1 0.052(P0.03/R0.18)、boundary-F1 **0.241**、**area-match 96.4%(pred/ref
1.037,超SAM3 OBIA 92%/FSDA 89.8%)**、各尺寸检出更高(>0.1ha 0.849/>0.2ha 0.866/小地块>0 0.759 vs SAM3 OBIA 0.46)、
over-seg 2.10(切碎FP多)。SAM3 OBIA 仅>1ha检出略优(0.88 vs 0.78)。
| 指标 | SAM3 ft+OBIA | DINOv3-Sat dist-peak |
| area-match | 92.0% | **96.4%** |
| boundary-F1 | 0.195 | **0.241** |
| instance-F1 | 0.031 | 0.052 |
| 小地块>0检出 | 0.46 | **0.759** |
| >1ha检出 | **0.88** | 0.78 |
**结论:纯DINOv3-Sat距离头 dist-peak watershed 是更强delineation线(单模型、area最优、boundary最高、各尺寸召回更高),
不需SAM3——呼应"1m-DINO为主、外部SAM3非必需"主线。** 两线互补:dist-peak=高召回密集小田、SAM3 OBIA=干净大田+少over-seg。
dual 完整重训(sam3_ft_dual_b2,存{seg,trans,dps})GPU0 进行中,待补 dual+OBIA 公平数字。

**[2026-06-06 更新10] dist-peak 参数调优 + 跨省泛化(4卡并行铺满):**
- **参数调优治over-seg(60cell扫→120全量确认):** min_dist/peak_thr/min_area 10/0.3/64(基线)→15/0.35/150→**20/0.4/200最优**。
  全量120: **instance-F1 0.052→0.096(近2×)、over-seg 2.10→1.91、area-match 96.4%→98.0%(pred/ref1.02,几乎完美,
  超SAM3 OBIA 92%/FSDA 89.8%)**、boundary 0.254。增大种子间距/阈值/min-area 明确治过分割。**dist-peak最优配置定:
  md20/pt0.4/ma200。**
- **ridge watershed(max(boundary,1-dist)高程)帮助小:** instance-F1 0.067/boundary 0.257/over-seg 2.06 ≈ 纯dist,
  因boundary头已饱和~0.55,ridge增益微。纯dist-peak足够。
- **长治跨省 dist-peak(120cell,c_1m_changzhi+changzhi DLTB 32万耕地图斑):** **area-match 95.0%(pred/ref0.95)、
  boundary-F1 0.338(>甘肃域内0.254!)、over-seg 1.82、instance-F1 0.069**。**跨省gap仅3%(域内98%→跨省95%),纯
  DINOv3-Sat距离头dist-peak跨省delineation泛化极强**,呼应"1m-DINO域不变"主线(分类跨省0.918-0.928,delineation
  跨省area95%同样几乎无gap)。对标FSDA西藏89.8%(域内),我们长治95%是跨省更严口径仍更高。脚本 dino_parcel_eval.py
  (--ridge/--dltb-parquet/--min-dist/--peak-thr/--min-area-px,glob无manifest区域)。dual ep1/4 val-F1 0.712训练中。

**[2026-06-06 更新11] dual 完整重训收官(Mac版sam3_finetune_b存{seg,trans,dps}):** ep1 0.712→ep2 0.727→ep3 0.749→
**ep4 0.770(mask-F1 best,>单源final 0.734,+0.036,数据增量确认有效;P0.72/R0.83)**。**dual(完整ep4)+OBIA 全量118:
area 91.2%(≈单源92%)、instance-F1 0.051(>单源0.031,SAM3系最高)、boundary 0.194、over-seg 1.43。** → dual 数据增量
OBIA后小幅提instance-F1,area持平,不质变。**最终 delineation 排名:dist-peak调优(area98%/inst-F1 0.096)>>SAM3 dual-ft+OBIA
(91%/0.051)≈单源ft+OBIA(92%/0.031);跨省 dist-peak 95%。纯DINOv3-Sat单模型dist-peak 全面最优,外部SAM3非必需(SAM3
价值在大地块+text-prompt灵活+少over-seg)。** 整轮(全部执行+空GPU路线)收官:dist-peak参数调优/ridge/长治跨省/dual完整
重训+dual&单源OBIA 全部完成,手稿§4.12+Table10更新。注:期间GPU2/3被同事huzi gemma-4训练占用,仅GPU0/1可用。

**[2026-06-06] 逐地块产品导出(dino_parcel_export.py):** 用最优 dist-peak 方案生成 GIS 矢量产品 —— 耕地区
dist-peak watershed 实例 + 非耕地类 connected-components → 全覆盖 7类实例图 → 每地块 GeoJSON(EPSG:4326)+GPKG
(parcel_id/class_id/label/label_en/rgb_hex/area_m2) + PNG(RGB|7类分类|地块黄边界) + legend.json。**area_m2 必须
用 像素数×pix_m²(4326下 geometry.area 是度²会错)**。4 cell 实测:620724_399 耕地密集 1076块(耕地692/园地153/水体158/
建筑62)、620105_308 黄河谷地 207块(林58/草33/水33/荒漠32/建筑18/园18/耕15)。产品在 .250 results/parcel_product/ +
Mac sidecar/parcel_product_demo/。可 QGIS/ArcGIS 直接打开。
- **tile 拼接接缝(方块线/横线)→ Hann 窗加权融合治本(2026-06-06):** ①先试 50% overlap(step=cs//2)均匀平均——仍有
  规则**横/竖直线**:根因="被1个tile覆盖区"与"重叠区(2-4个tile)"交界处覆盖数cnt跳变→平均值不连续=接缝线。②正解=
  **Hann 窗加权**(infer_heads:每tile输出×`np.outer(hann,hann)`中心权重1边缘→0,加权累加/权重和,floor 1e-3防角落除0)——
  重叠平滑过渡、无cnt跳变,接缝彻底消失(梯田弧形等高线边界无横直线验证)。tiled inference 标准做法,推理时治本非后处理。
- **梯田+耕地案例批次:** 梯田 c_1m_terrace(2000 cell,黄土高原等高线梯田)→ parcel_product_terrace(4 cell,梯田正确
  识别为耕地+村庄红+渠蓝,边界跟等高线条带无方块);耕地密集 c_1m_lc top(620724_595耕地811/620721_313耕地853/
  624001_351/620421_486)→ parcel_product_cropland。共12 cell 产品在 Mac sidecar/parcel_product_demo/。

**[2026-06-06] 榆中县(620123)整体分析 — 两影像路线 → GeoParquet:** 用最优 dist-peak 模型(dino_v3_bdd,Hann融合)
对榆中全县逐地块分析,两条影像路线对比:
- **路线1 下载影像 c_1m(~1m,95 cell):** 55278 地块, 耕地+园地 39287块/155.1 km², 全类 316.2 km²。
- **路线2 本地影像 hires_jpg(~2m,80 cell,esri+google tif):** 41616 地块, 耕地+园地 29191块/151.4 km², 全类 316.9 km²。
- **关键: 两路线耕地面积高度一致(155.1 vs 151.4 km², 差2.4%)/全类面积一致(316 km²)**, 尽管影像源/分辨率/cell数不同
  → **dist-peak 模型对影像源稳健(交叉验证)**; 1m 比 2m 地块更细(55278 vs 41616)。GeoParquet(EPSG:4326,
  列 gid/parcel_id/class_id/label/label_en/rgb_hex/area_m2/cell)。
- 实现: dino_parcel_export.py 加 `--tif-dir`(本地esri+google tif→x6,load_tif_pair,3857→4326)/`--prefix`(整县glob)/
  `--region-out`(合并区域GeoParquet)/`--parquet-only`。产品 .250 results/yuzhong_{c1m,tif}_region.parquet +
  Mac sidecar/yuzhong_product/。

**[2026-06-06] 榆中县三项收尾分析:**
- **③ 县级对象级精度(95 cell vs DLTB 620123,dist-peak最优 md20/pt0.4/ma200):** area-match **90.6%(超FSDA 89.8%)**、
  boundary-F1 0.286、instance-F1 0.058、over-seg 1.91。真实县级产品精度确认(榆中含黄土山地+城镇,比甘肃混合test 98% 略低
  但仍超FSDA)。
- **② 两路线空间一致性:** 逐cell耕地面积相关 **r=1.000**(完美线性一致);全县 cropland 148(1m) vs 145(2m) km²(差2%)、
  各类分布一致(草地70/74、建筑62/61、林地19/21);公共80cell耕地+园地2m比1m偏大~23%(2m地块标定偏大的分辨率效应)。
  → dist-peak 对影像源稳健。
- **① 区域预览** yuzhong_preview.png(两路线视觉一致,采样cell分布同、分类格局同)。
- **注:** c_1m 榆中=95个采样2km cell(~380km²采样点)非连续全县(3300km²);连续全县需下载器下全县瓦片。
  脚本 yuzhong_analysis.py/verify_yz.py。

**[2026-06-07] parcel_dist 部署后端 — 最优 dist-peak 接进 GUI/CLI:** 新后端把最优 delineation 线接入下载器分类:
- `sam3_classify/parcel_dist.py`: read GeoTIFF → x6(单源dup) → DinoV3FreqUNetBDD 距离头 → infer_heads(Hann融合,无接缝)
  → build_idmap(dist-peak watershed 耕地 + connected-components 其他类) → 全覆盖7类 → **GeoParquet(默认)+GeoJSON+GPKG+
  class raster+legend**。
- infer.py dispatch + __main__ choices 加 `parcel_dist`; classify.rs `default_parcel_dist_weights()`(~/D/cropland_dino/
  parcel_dist.pt) + backend match + is_trained。权重 dino_v3_bdd→Mac。
- **验证(榆中 620123_504 tif, cuda):** stage流完整, done 含 label_parquet, 406地块/7类。GUI/CLI: `--backend parcel_dist`。
  (前端模型下拉加 parcel_dist 选项=待办 UI 小改。)

**[2026-06-07] ① 榆中连续区域全图(下载器→parcel_dist→GeoParquet):** 下载器 batch 下榆中县城+农田连续 5×5 网格
(620123c, z17 esri, ~12×12km, 25 cell 密铺, 167s) → cell-wise dino_parcel_export --tif-dir(google缺则dup esri)
→ **连续 GeoParquet 13519 地块**(耕地9428/水体1091/建筑930/草地883/林地657/园地529)。
- **单大图路线弃:** rasterio.merge 25 tif → 10254×10350 单图 → parcel_dist,但 watershed+CC on 1.06亿像素单核 >34min 太慢。
  改 cell-wise(各 cell Hann 融合无 tile 接缝,密铺连续,仅 cell 共享边轻微痕)。
- 预览 yuzhong_continuous_preview.png: 县城红+农田绿+河蓝+山林深绿,连续覆盖。产品 sidecar/yuzhong_product/
  yuzhong_continuous_region.parquet。脚本 merge_yz.py/plot_yz_cont.py。
- **教训:** 大区域连续 delineation 用 cell-wise(GPU推理快+Hann无tile接缝) 优于单巨图(CPU watershed 1亿像素瓶颈);
  cell 边痕可后处理 dissolve。

**[2026-06-07] 连续大图消 cell 硬接边 + 线条平滑(用户反馈):**
- **① cell 硬接边 → 单大图统一 watershed:** cell-wise 各 cell 独立 watershed→cell 边截断成硬直线。改用 rasterio.merge
  拼单大图(10254×10350) → parcel_dist 一次统一 watershed(连续,无 cell 边)。**降采样加速:** dist_peak_instances 加
  `downscale`(dist/crop/bnd 降 N 倍→watershed→NEAREST 上采样),parcel_dist 大图(max>5000)自动 downscale=4(watershed快~16×)。
- **② build_idmap O(n×像素)瓶颈 → 全向量化(真正治34min慢):** 耕地 cls_of 原 per-pid `idmap==pid`(数千×1亿)→
  **bincount 加权**(clsprob[1]/[2] 一次算所有 instance);其他类 CC 原 per-lab `cc==lab`→**connectedComponentsWithStats
  +向量化赋值**。消除所有 per-instance/per-label 全图 mask 扫描 → delineating 从 >34min 降到几分钟。
- **③ 线条平滑:** smooth_geom(Chaikin 角点切割,栅格阶梯→平滑曲线,拓扑安全) + Douglas-Peucker 简化。args --downscale
  /--smooth-iters。
- **结果:** 榆中连续 ~12×12km 单图,12961 地块,**无 cell 接边、边界平滑、连续全覆盖**(yuzhong_cont2_preview/zoom.png)。
  parcel_dist 部署后端同步受益(大图自动 downscale+平滑)。

**[2026-06-09] Frame Field Learning (栅格转矢量学习型方案) — 验证成功:** 用户问"训练好的栅格转矢量大模型"——
答=Frame Field Learning(Girard CVPR21),用 DLTB 微调。
- **实现:** DinoV3FreqUNetBDDF(frame field head 输出复多项式 c0/c2) + train_framefield.py(DLTB边缘切线 GT=
  dist-transform梯度旋90°→û²; FFL align loss |f(û)|²+|f(iû)|²; warm-start dino_v3_bdd head-only 0.15M; spawn
  workers避CUDA-fork) + ff_polygonize.py(边snap到学到的frame field方向 edge-regularization)。
- **结果(620724_399):** 931地块, **vertices/parcel mean 80.6(后处理Chaikin)→18.9, max 78481→103**, 规则直边
  条田、无波浪锯齿(预览 ff_poly_preview.png)。frame field仅3ep(align 1.290→1.245趋平,head-only粗)已显著规则化。
- **关键修复:** infer_heads autocast改条件(Mac MPS/CPU兼容)/parcel_dist dev一致(model+data同device)/BDDF返回4-tuple
  infer_heads取前3/ff detach()/_line_intersect近平行飞点clamp 40px(否则extent爆到1°×1°)。
- **小问题待修:** ff_polygonize per-instance contour 间有白gap(regularize收缩, 可gap fill); frame field联合训
  decoder(multi-task)会更准→更规则。**栅格转矢量学习型(FFL+DLTB)验证可行,polygon规则度远超后处理。**
- **服务器:** .250 曾宕机~1hr(uptime重置), 恢复后续跑; 所有进展无丢失。
