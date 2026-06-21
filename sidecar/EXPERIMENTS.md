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

**[2026-06-10] FFL 联合训完成 + 结论:** train_dino_7class --frame-field 多任务联合训(cls+bnd+dist+frame, 6ep,
dino_v3_bddf, warm-start dino_v3_bdd, OA 0.804 保持没退/macroF1 0.649)。auto 重测 ff_polygonize:
**联合训(6ep) mean 19.0 max 257 ≈ head-only(3ep) mean 18.9 max 103** — 都远优于后处理 mean 80 max 7.8万。
- **关键结论: FFL polygonization 的规则度对 frame field 精度不敏感**(edge-snap 只需方向粗对, 3ep 粗 frame field
  已够), 联合训 polygon 不比 head-only 更规则。联合训的价值 = **一个模型同时出 cls+bnd+dist+frame**(部署友好)。
- loader 瓶颈: frame GT dist-transform(getitem) + 多时相 NDVI 磁盘IO, ~37min/ep(GPU~3.5%饿); 减规模/预生成 GT 可加速。
- ff_polygonize 剩白缝(per-instance contour 收缩, 非耕地类少), 待 gap fill 治。
- **FFL(栅格转矢量学习型, DLTB监督)完整验证: polygon顶点 mean 80→19, 直边规则无波浪。** 脚本 train_framefield.py
  (head-only) / train_dino_7class --frame-field(联合) / ff_polygonize.py(edge-regularization)。期间.250宕机1次恢复。

**[2026-06-10] FFL 三件套收尾(全做+论文):** ① ff_polygonize 顶点位移 clamp 40→8px(局部拉直波浪但贴住共享边界)
→ **白缝基本消失**(对比图田块连片, 1010 parcels=后处理同数=全覆盖, mean 19.8)。② parcel_dist 部署集成:
模型类→DinoV3FreqUNetBDDF(超集,旧BDD权重兼容), 检测 ckpt 含 frame_field_head 权重时自动走 polygonize_ff
("ff_regularized":true), 否则回退 simplify+Chaikin; 联合训权重(dino_v3_bddf, cls+bnd+dist+frame 四头一模型)已传
Mac parcel_dist.pt — **GUI parcel_dist 后端自动输出 FFL 规则矢量(GeoParquet)**。端到端验证榆中tif: 380地块/7类/
ff_regularized:true。③ 论文: rse_manuscript §4.12 加 FFL 小节(DLTB边缘切线监督 frame field, 顶点 mean 81→19/
max 7.8万→~10², 规则度对 frame 精度不敏感=3ep head-only 已够, 联合训价值=四头一模型部署; 首个用国土调查矢量
监督 frame-field 全类地物(7类)矢量化, 耕地为其中最准类) + Girard CVPR21 引用, pandoc 同步 docx。

## 2026-06-12/14 — 无缝县级矢量化定稿 + CLI 通用化无缝集成 + 神池跨省 e2e + C-RADIO 骨干对照

**无缝县级矢量化(榆中定稿 SMOOTH3):** FFL 帧场矢量化**县级 coverage 被否** —— 逐实例正则使共享边各画一次 → **6-14% 结构性重叠**(撑面积)+ 过直/梯田曲线打成直弦(snap_deg 0→90 重叠不变=非snap/clamp问题, 是逐实例架构)。正路 = **全县单图统一 watershed**(7.3Gpx 整图爆内存 → **/4 全局累加器**: 滑窗1m推理→降采样/4→Hann累加到全局/4网格 25409×17943→一次 build_idmap(ridge)→无块无缝 partition)→ **全县整体拓扑保持矢量化+曲线平滑**(coverage_simplify(tol≥1像素步4.78m, 5m甜点)去/4阶梯 → 全县一次 topojson(简化后~20GB不OOM, 539k arcs, 证明OOM先简化即可解非死路)→ 每条共享 arc Chaikin 角切端点固定 → **零重叠零空白 by construction**)。Chaikin 收敛: 3次到二次B样条极限(3→4 顶点翻倍不变顺), 推荐 **Chaikin3**(角切固有内缩~2%全在外边界, 裁县界后面积-0.0%)。**标准后处理 `postproc.py`**: eliminate_slivers(细长度判 w=area/peri<2m+PP<0.3 → 并入最长共享边邻块, 非删)+ fill_gaps_holes(county.difference 检图斑间空白+未填空心洞→多数票填)+ fix_invalid。成品 **yuzhong_SMOOTH3(Chaikin3+sliver, 100906 地块, 面积 3166.4=三调620123 -0.0%, 零重叠 6e-8%, all-valid)**。脚本 yz_global_ffl.py / yz_smooth2.py / yz_postproc.py。**关键 bug**: topojson 在 island/hole 边界留单引用 arc, Chaikin 后两份独立移动 → 1.39% 真重叠, coverage_simplify(tol=0) snap 治不了 → 改 **STRtree 逐重叠对挖重叠区归大块**(exact 无缝)。

**CLI 通用化 + 无缝集成(取代逐cell):** `parcel_pipeline.py`(区域无关: --mosaic/--cells-dir/--weights/--backbone/--boundary/--downscale/--smooth-iters/--utm/--device)+ `postproc.py`(run_postproc 标准末步)。**设备可移植**(--device auto/cuda/mps/cpu; 非cuda 单设备整图滑窗 fp32 不开autocast; cuda 多GPU行带保留)+ **自带拼接**(--cells-dir rasterio.merge)。**CLI `src-tauri/cli.rs` 训练后端(parcel_dist/cropland/parcel_bh/parcel/landcover)下载后改调一次 parcel_pipeline → `<out>/county_seamless.parquet`, 取代逐cell有缝版**;单图 slic/sam3/dino 保留逐cell。管线 env = `IMG_PIPELINE_PYTHON`(默认 ~/miniconda3/envs/py312, sam3 env 缺 topojson)。Mac MPS 2×2格实测跑通+cargo build/test 全过。commit 1654928/5e126d2/e228b22。

**神池县(山西 140930)跨省 e2e 能力测试(用户要求只测能力、不做精度验证):** CLI `batch` 下载 **457格**(县界∩0.02°格, OSM 县界, 52min/0失败)→ parcel_pipeline(.250 CUDA)拼接 3.4Gpx → 4卡推理 16min → 全局watershed → Chaikin3 → 裁神池县界(--boundary 直接吃裸 Polygon geojson)→ postproc → **50096 地块无缝**(sum==union 零重叠, 100%覆盖县界, all-valid, 1483.4km²)。逐类 耕41.5/草32/林20/建5.5/水0.7/荒0.2/园≈0(神池无果园)= **完全吻合晋北黄土丘陵**, 无崩溃/异常类。**整条新无缝工具链跨省一遍跑通**。暴露并修 **多GPU bug**: worker `log()` 用 %d 格式化字符串 gpu号(--gpus 0,1,2,3 split 成str)→ 4卡 worker 全 TypeError 崩(榆中走整数/--mosaic 没踩到)→ 改 %s(e228b22)。Chaikin+topojson 单线程 ~44min 是瓶颈。产物 sidecar/shenchi_product/(全县+4放大大图)。

**C-RADIOv4-SO400M 骨干对照(NVIDIA AM-RADIO v4, SigLIP-SO400M 蒸馏, ~430M):** 三层结论, 全部"换骨干无增益、训练比骨干重要"。**注: 强性能均来自用 c_1m 微调后**, 原始/零样本不具备:
- **零样本(原始 C-RADIO, 无微调)**: 开放词表; C-RADIO 释出 ckpt 无 SigLIP adaptor 头(adaptor_names=null)→ 退用其蒸馏 teacher **SigLIP2-SO400M** text-prompt(诚实标注): 梯田耕地 pixel-F1 **~0.62 过预测**, << 微调后, 低~0.30 → **裸用基础模型不够**。
- **二分类微调后(同口径配对: c_1m 同数据/FreqFusion/头/parcel_eval)**: pixel 0.868vs0.870 / area-F1 0.933vs0.934 / 小地块count 0.748vs0.751 / 0.5ha-MMU 0.931vs0.936 —— **全 ≤0.005 噪声内打平**, C-RADIO 无一胜出(更大更慢双pass)。6ch 接法=双 RGB triplet 各跑一遍 concat(C-RADIO 无6ch patch-embed)。
- **7类榆中产品 yuzhong_SMOOTH3_RADIO(微调后, 89181地块, 面积±0% 零重叠 all-valid)**: 与 DINOv3-Sat 县级**空间逐像素一致率 88.5%**(草/耕/林主类 81-96%一致); C-RADIO 耕地→草地混淆略重(耕-87/草+133km²), **诚实归因 = 7类只单段10ep warm-start vs DINOv3-Sat 三段 bdd→bddf→bddf_enh 训练预算差, 非骨干能力**。
- 脚本 train_radio_1m.py / train_radio_7class.py(RadioFreqUNetBDDF)/ yz_global_radio.py; 权重 radio_1m_gdlxff/best.pt + radio_v4_bddf/best.pt(0.7166 vs DINOv3-Sat 0.7538)。坑: dist-transform GT 在192核炸线程(OMP/MKL_NUM_THREADS=2 修, GPU 0%→99%)、eval 慢(--eval-cells 40)、假 DONE 误触发 Phase2。

**RSE manuscript 更新(`~/CODE_BLOCK_DNDC/cropland_1m_parcel/docs/rse_manuscript.md`):** §4.13 整段重写=无缝拓扑有效 coverage(取代旧 per-cell+FFL 产品, Table10 真实逐类 vs DLTB620123)+ §4.12 FFL 衔接(逐实例→县级拓扑保持)+ Abstract/§1.4贡献5/结论加无缝县级部署 + **表号统一 1-11**(原 Tibet跳7/文献6落尾)+ Fig.7 梯田骨干对比(零样本/训练后RADIO/本文 三列)/ Fig.8 榆中县级 + §4.6 配对 null 结果 + refs 17 AM-RADIO/18 SigLIP2 + Data&code availability 补实。commit dfbea8d/9054e28/696c2ba/1dd67a3/092f78d/858eaeb/bf6344e。fig9(榆中7类C-RADIO对比)交付未入正文(训练段数 caveat)。

## 2026-06-15/16 — "悬空线段"伪影根治(建筑路网巨斑 giant-skip)+ 伪影vs真梯田诊断纪律 + CLI自动清理 + 神池对齐

**用户报最终 geoparquet 满是"悬空细线段"。诊断=两类, 治法不同:**

**① postproc 强化(治 Chaikin 节点微伪影, commit 0c5520d/2e77f7f):** 诊断 Chaikin 平滑在拓扑节点留两类微伪影 —— ~8845 个**细 sliver**(w=area/peri<0.5m, 渲染 boundary 时细长闭环=悬空线段)+ **108282 个近零面积退化 interior ring 微洞**(median 0m², 亚像素看不见但脏)。原 `eliminate_slivers` 只清 136。强化: `is_sliver` 改 **细(w<1.5m)且小(area<100m²)**(面积上限护真路/长梯田); `eliminate_slivers` 用 STRtree bbox 候选 + 向量化邻接(避巨斑 boundary∩boundary 慢)并入最长共享边邻块; 新 `drop_tiny_holes`(删 area<30m² 退化环, 留嵌套/大洞); `run_postproc(skip_gaps=)`(无缝输入跳 fill_gaps 全量union, 防误填路网合法内洞致面积虚高+90km²); `standardize` 末加 make_valid。**已进 `parcel_pipeline.run_pipeline`→run_postproc=CLI县级自动清理**。

**② 上游根因=建筑路网巨型多边形(commit 26b30fc/731c34e):** 残留细 sliver 集中在**建筑(道路网)被连通域标号成的单个巨型多边形**(榆中 maxV **510万顶点**/上千洞=被路网圈的田)边界 —— Chaikin 逐弧平滑这种超复杂路网边界, 在无数交叉节点**狂产细楔形 sliver**, 合并又在其边再生(死循环), 清不动。修=`smooth_coverage._giant_adjacent_arcs`: 邻接**线状网络型巨斑**的 arc **跳过 Chaikin 保直边**(路本就直边), 其余照常平滑。**判据迭代(关键)**: 初版"仅顶点≥5万"**误伤草地(53万顶点)/林地团块** → 田块↔草地/林地边被跳平滑留折角, 用户报"反而不如SMOOTH3平滑"; **修正为 顶点≥5万 且 shape_ratio(周长²/面积)≥10万**(类无关线状判据: 榆中建筑 shape_ratio=403850线状 vs 草地12k~58k/林地2.9k紧凑, 拉开~7x)→ 只路网命中, 草地/林地团块边重新Chaikin成曲线, 跳过arc 51144→26981。**榆中建筑 maxV 510万→69万, 楔形 sliver→0, 草地/林地边平滑, 路网直边。**

**③ "伪影 vs 真地物"诊断纪律(用户逼出的关键认知):** 残留 ~4500 细 sliver **不是伪影、是真地物, 不该删**。三步验证: (a) 类别分布 w<0.5 sliver = **耕地65%/草地16%/林地8%/建筑(路)仅6%**; (b) **叠1m影像**看耕地 sliver 落在真梯田等高线上(一串绿点沿梯田边); (c) **同类合并仅-8%**(底层就是细的、多无同类邻居), 跨类合并=毁真地物。**铁律: 真伪影(Chaikin楔形)清掉后剩的细线是 1m 忠实捕捉的真窄梯田条, 嫌出图碎用显示层 MMU(area<300m²隐藏=0.028%面积)不改数据。** 神池反证: 路网稀→giant-skip后 w<1 sliver 10007→**4(近零)**; 榆中真梯田多→残留6096=真地物。

**④ 神池(140930)giant-skip 对齐(同口径):** 同命令重跑(唯一变量=代码, 推理产物57354polys/137763arcs逐项一致)。建筑 maxV **390万→473k(-88%)**、w<1 sliver **10007→4**、invalid 1→0、面积1483守恒、零重叠、逐类±0.2%对齐榆中。`shenchi_FINAL.parquet`(39215polys, 比旧小因巨斑顶点不再翻倍)。

**产物/口径:** 榆中终版 `yuzhong_FINAL.parquet`(=v2b, 73325地块, 591MB, 草地/林地边曲线+路网直边+楔形0, 面积3166.2/-0.005%/valid/零重叠); 神池 `shenchi_FINAL.parquet`。giant-skip+postproc 进 main→**CLI 县级自动产出此水平**。**论文 §4.13 补句**(线状网络巨斑跳过平滑+残留=真梯田保留+MMU显示, commit 81dac44)。PIPELINE.md + 记忆(terrace-band)沉淀。**性能坑**: 建筑线状巨斑重Chaikin后420万顶点, topojson/resolve/clip/union 各阶段慢, 榆中v2b端到端161min(免推理)→ 日后值得对建筑巨斑矢量化后单独几何简化提速。新脚本 `diag_giants.py`/`yz_regen_v2b.py`/`clean_products.py`。**工程纪律**: 别用 until-loop 后台轮询(harness杀前台sleep, exit144)→ setsid+sleep读文件; agent 别派会空转的 Monitor 子任务(神池agent卡死循环); session限额掐断agent但服务器setsid进程独立续跑。

## 2026-06-17/18 — ONNX 导出(部署可移植)+ 数值 parity 硬化 + 分发打包(零改生产代码)

**需求:** 把"下载→拼接→推理→landuse polygon"生产管线打包成**可分发的包**(目标 Linux + 团队可复现 + 论文 code-availability),**不动任何生产代码**;推理段允许**新增** ONNX 后端(route A)以脱 torch/transformers 依赖。

**生产模型与权重(核准,纠正前期误判):** 生产模型 = `DinoV3FreqUNetBDDF`(DINOv3 ViT-L/16 sat493m 骨干 + FreqFusion 解码 + 四头 cls/bnd/dist/frame;in_channels=11, num_classes=9, unfreeze_last_n=4)。**实际部署权重 = `/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt`**(三段训练 bdd→bddf→bddf_enh 终段, val 0.7538);铁证 = `run_shenchi_final.sh` 的 `--weights` + `yz_global.py:14`/`yz_global_ffl.py:20`。⚠️ **权重溯源坑**:`dino_v3_bddf_enh/best.pt` 与 Rust CLI/GUI 默认的 `parcel_dist.pt`(= `dino_v3_bddf/last.pt`,第二段,md5 `7a98140b…`)**不是同一个**;两个终版产品由 enh/best.pt 生成,**不是 parcel_dist/last.pt**(签入的 yz_pipeline.py 默认值已陈旧)。

**生产运行环境(核准):** conda **base**(`~/miniconda3/bin/python`):py3.13.9 / torch 2.9.1 / transformers 4.57.3 / rasterio 1.4.3(gdal 3.9.3)/ geopandas 1.1.0 / shapely 2.1.1 / topojson 1.10 / opencv-python-headless 4.13.0.92 / scikit-image 0.26 / scipy 1.16 / numpy 1.26.4 / pandas 2.3。(py312=torch2.5.1+onnxruntime 的**导出** env,非生产。)

**ONNX 导出(`sidecar/export_onnx.py`):** 导出 448 瓦片前向 → `results/parcel_enh.onnx`(1.22GB,**纯 onnxruntime 加载,无 torch/transformers/mmcv**)。签名 `x[1,11,448,448] f32 → cls[1,9,224,224] logits / bnd[1,1,224,224] / dist[1,1,224,224]`(头原生分辨率 crop/2=224;第四头 frame_field 经 ThreeHead wrapper 丢弃,推理本就忽略 o[3])。softmax/sigmoid + interpolate→448 + cv2 /N resize + Hann 累加仍在 Python。

**⚠️ transformers 版本陷阱(本次最重要、论文级复现警告):** 加载 `dino_v3_bddf_enh/best.pt`:**transformers 4.57.3(生产 base)→ 914/914 全加载 ✓**;**transformers 5.9.0(py312)→ 仅 98/914**(5.x 改了 DINOv3 backbone 参数命名,checkpoint 骨干 key 全对不上)→ **微调的解冻骨干被静默丢弃**、退回 from_pretrained 基座骨干 → 输出大变(softmax 差 ~0.5、argmax 仅 **88.5%** 一致)。**必须 pin `transformers==4.57.3`**(`environment.yml` 已固定);已给 `export_onnx.py` 加**加载完整性硬告警**(<914/914 直接 WARNING,防再静默烤错权重)。先前误判为"torch 版本漂移",实为 transformers 版本(已用 eager/SDPA=9e-7、原 CARAFE/exact=0.0 排除其余因素)。

**数值 parity(硬化,关键方法学 + 一次自我纠错):** 初版在 py312(transformers 5.9.0)导出,parity 6.26e-06 PASS —— 但那是"onnx 忠实复现了一个 **98/914 残缺加载**的 torch 模型"的**自洽假象**(garbage-in→garbage-out)。**在生产 base(transformers 4.57.3 / torch 2.9.1 / 914/914)重导**,5 独立随机输入内部激活最坏 **|Δ| = 4.59e-06 → PASS**,才是忠实于生产的 ONNX(`parcel_enh.onnx` sha256 `48c36c2c…`)。双教训:① parity **必须多输入取最坏**(单输入 dummy 值被 legacy 追踪器烤成常量,曾骗出 3e-05 假 PASS);② **必须在生产同款 env 导出**,否则 parity 自洽却对着错权重。

**端到端验证(纠正后):** 小样(榆中 mosaic 中心 2000×2000)route-A onnx 路 vs 生产 torch 路:onnx **44 地块/3.710 km²**(草 3.312/耕 0.308/水 0.030/林 0.027/建 0.034)vs torch **46 地块/3.676 km²**(草 3.296/耕 0.289/水 0.030/林 0.027/建 0.034)—— 逐类吻合,残差 ~1% 来自 onnxruntime-CPU-fp32 vs torch-CUDA-bf16(非正确性)。错版 py312 onnx 当时给"耕 0.902/草 2.715"(明显错)→ 纠正后两路一致。**工程纪律**:`pip install onnx/onnxruntime` 会贪婪顶高 numpy(base 1.26.4→2.4.6,破 rasterio/shapely ABI)→ 已 `pip install numpy==1.26.4` 复位,base 全依赖 import 复检 OK。

**导出配方/踩坑(可复现必读):** ① 骨干 `attn_implementation="eager"` 加载(SDPA 难 trace,与 eager 数学等价);② monkeypatch forward 把 `int(round(N**0.5))` 换 python 常量 P=28/D=1024(追踪期 N 是 Tensor,`round` 报错);③ **CARAFE/FreqFusion fallback 的 `F.interpolate(mode='nearest')` → `repeat_interleave`**(torch 与 onnxruntime 的 nearest 索引约定不同 → 不修内部差 **~3e-2**,头号坑;整数上采样数学等价,**仅导出进程 monkeypatch,不动 FreqFusion.py**);④ legacy `torch.onnx.export` opset17(**`dynamo_export` 在 torch2.5.1 报 OnnxExporterError,死路**;torch≥2.6 默认可能走 dynamo → 用 `inspect` 检测并传 `dynamo=False` 强制 legacy;torch2.9.1 的 legacy 还需 `pip install onnx`);⑤ fp32 CPU 导出,parity torch-fp32 vs onnxruntime-CPU。

**route-A ONNX 后端(新增文件,零改生产代码):** `parcel_pipeline.py` 模块级 torch-clean(torch import 全在函数内)、`postproc.py` 全 torch-free → 新增 `sidecar/onnx_backend.py` 可 **import 复用** make_hann/_read_grid_meta/_finalize_acc/vectorize_idmap/smooth_coverage/clip_to_boundary/load_boundary/mosaic_from_cells + postproc.run_postproc;仅 **verbatim 复制** enhance6/norm6/build_idmap/dist_peak_instances/_softmax/_sig(函数体无 torch,但宿主文件模块级 import torch 故不可直接 import)。推理用 onnxruntime 替 torch 前向,其余无缝矢量化链原样复用 → 输出与 torch 路径数值一致。

**分发打包(`sidecar/deploy/`,均为新增文件):** 主形态 **Docker**(`nvidia/cuda:12.1` + micromamba)+ **`environment.yml`(真源**,按 base 生产版本固定 + onnxruntime;**刻意不装 mmcv** 以保 FreqFusion 纯 torch fallback 数值不变);权重**外置**(HF Hub 主 / GitHub Release / 内网 174 光纤),`checksums.sha256` 把关;**torch 路径=参考/论文数值,onnx 路径=轻量等价**(已 parity 验证)。conda-pack 仅作内网同构机离线兜底。GDAL 版本错配是头号复现风险 → Docker 烤一份 + conda-forge 出 GDAL,严禁 pip-gdal 混装。

**两个终版产品(核准统计,论文用):**
- `yuzhong_FINAL.parquet`(榆中 620123,甘肃):**73,325 地块**(读 parquet 实测;PIPELINE.md §⑤ 的 74177 系旧 typo),591MB,EPSG:4326,7类,全 valid,零重叠。**总面积 3,166.24 km²(UTM48N)= DLTB620123 −0.005%**。逐类(数/km²):耕 42036/1016.7、草 7942/1403.5、林 8056/429.0、建 10380/256.9、水 3430/44.2、园 706/11.4、荒 775/4.5。
- `shenchi_FINAL.parquet`(神池 140930,山西):**39,215 地块**,343MB,7类全 valid 零重叠,100% 覆盖县界,~1483 km²(县口径)。**跨省能力验证,非精度声明**(无山西 DLTB)。逐类:耕 23585/621.3、草 4886/479.1、林 5771/299.9、建 3676/82.5、水 800/10.2、荒 478/3.3、园 19/0.2。

**RSE manuscript(`~/CODE_BLOCK_DNDC/cropland_1m_parcel/docs/rse_manuscript.md`)待办:** §4.13/Table10/Fig.8/Abstract/结论的 SMOOTH3 旧计数 **100,906 → 73,325**(面积逐类几乎不变:耕 1016.7 vs 1016.5、草 1403.5 vs 1403.6、总 3166.2/−0.005%;计数降因路网巨斑顶点不再翻倍 + Chaikin 楔形并掉);§4.13 giant-skip 判据细化为"顶点≥5万 且 shape_ratio≥10万"双判据;ONNX 导出/parity/打包入 Data & Code Availability(部署/复现细节、非新科学结论,一行提及,主记于本台账)。

**部署补充(2026-06-18,本会话续):** ① **复现确认**:`transformers 4.57.3` 安装于 2026-01-05,**早于**两终版产品(06-16)→ 产品确用 4.57.3(914/914 正确加载),当前 env 可复现,论文钉 `transformers==4.57.3`。② **onnx 路运行期实测 torch-free**:屏蔽 torch import 后整管线端到端跑通(`torch in sys.modules=False`,小样 44 polys/3.7km² 与 torch 路逐类一致)→ 部署 env 可**不装 torch/transformers/mmcv**。期间修一处 lazy-import 真 bug:`onnx_backend` 原借 `pp.default_classes()` 会 `from dino_parcel_export import CLASSES` 拖 torch → 改本地 CLASSES。③ **不用 Docker 直接布设套件**(`sidecar/deploy/`,新增):`environment-onnx.yml`(精简 env 无 torch)+ `install_bare.sh`(micromamba 无 root 一键装+取权重+校验+烟测)+ `serve.py`(FastAPI 异步作业 HTTP 推理服务:POST /infer→GET /jobs/{id}/result)。**CUDA 可选**(CPU 能跑;整县慢,建议 `onnxruntime-gpu`,仍无需 torch)。onnx 路只需 `parcel_enh.onnx` 一个权重(自包含)。④ **后处理 GPU 加速结论(诚实)**:县级瓶颈是后处理(榆中 v2b 免推理 161min CPU 单线程)非推理(神池 16min/4卡)。后处理大头 = GEOS/shapely 计算几何(topojson 拓扑/共享弧 Chaikin/resolve_overlaps/intersection/make_valid)**无成熟 GPU 等价库**(cuSpatial 不覆盖拓扑保持简化+多边形叠加+修复);唯一可 GPU 是 build_idmap 栅格段(EDT/watershed/CC/peak,可用 .250 现成 `rapids-25.06` 的 cuCIM+cupy)但非大头。**真正提速在 CPU 侧**:逐特征多进程铺核 + 建筑巨斑矢量化后单独几何简化(420万顶点是 161min 主因)。机器按 CPU 核多+内存大配,GPU 只值得加速推理。

**长治市(山西 1404,跨省)全市连续无缝 run + 两个 city-scale 修复(2026-06-18,进行中):** 整体管线做长治市全市 landuse polygon(地级市 ~13864km²,神池 9×/榆中 4×)。边界取 store .174 `长治边界缓冲.shp`(3km缓冲)Mac 中转→.250,`make_changzhi_grid.py` 生成 4216 格连续网格(0.02°∩缓冲)+ 精确市界 geojson(负缓冲还原 14152km²)。Mac Rust `imagery-downloader batch` 下 4216 格(esri z17,4.4h/0失败/8.8GB)→ rsync 推 .250。**两个城市级坑**:① **`mosaic_from_cells` 4GB 经典 TIFF 崩**(`TIFFAppendToStrip: Maximum TIFF file size exceeded`,长治 mosaic 26.8Gpx>4GB;县级没踩到)→ **生产代码修复**(用户授权):`parcel_pipeline.py:100` 加 `bigtiff="IF_SAFER"`(>4GB 自动 BIGTIFF,县级照旧);当前 run 已启动故临时用 `gdalbuildvrt` VRT + `--mosaic` 绕过。② **多卡内存爆**:`_band_worker` 累加器存 `/dev/shm`(RAM),长治 /4 ≈80GB/worker → 4 卡=数组320+shm320≈640GB 爆 503GB RAM(/dev/shm 也~250GB)→ 用 **`--gpus 0,1`(2卡)**峰值~322GB 进 422GB 可用,`/4` 不变保质量。推理 ~5-6h(2卡)+ 后处理 ~10h。脚本 `make_changzhi_grid.py`/`run_changzhi_final.sh`(新增)。成品 `changzhi_FINAL.parquet` 出后补逐类统计。

**长治市成品定稿(2026-06-19,clean global)+ prefecture 尺度三道墙完整结论:** 地级市(~14152km²,26.8Gpx)走全局法撞**三道 city-scale 墙**:① mosaic>4GB(经典TIFF)→ `bigtiff=IF_SAFER`(生产已修);② 推理累加器(每worker全图/4≈80GB,4卡爆503GB)→ `infer_global_memmap`(band-local切片+磁盘memmap,逐位一致,生产已并入,4卡满跑);③ **topojson(全市445k多边形+路网巨斑的拓扑复杂度)即便 tol=15 仍 OOM** —— **光调tol解不了,是 prefecture 硬墙**。**分幅(yz_blocks方案B质心)也否**:城市级块边bbox-core重叠致~14%重复计数(榆中没暴露/长治暴露),且修它的resolve_overlaps在1.77M地块卡死14h(县级函数扛不住)。**正解=干净全局跳过topojson**(`run_changzhi_global2.py`/`run_g2.sh`):全局idmap是分区→`vectorize_idmap + shapely.coverage_simplify` 已无缝零重叠,跳过只磨曲线的topojson/Chaikin(正是OOM步)→边为简化折线(非曲线,城市级够用)。**成品 `changzhi_FINAL.parquet`:293,788地块,14,111.7km²(市界14,152.6,−0.29%),全valid,重叠sum/union=1.0(零重叠完美无缝,对比分幅1.136),7类(林6954/耕3594/草1703/建1379/水287/园168/荒27),EPSG:4326**。memmap 4卡推理 + idmap.npy+cls_of.pkl 存盘。**工程坑(耗多轮)**:`pkill -f <脚本名>` 自匹配杀启动shell→用 bracket trick `[r]un_xxx`;前台Bash含sleep/timeout被harness拦→run_in_background或远端sleep;spawn脚本执行体须在`if __name__=="__main__"`内。**省级+方案**:全局idmap内存到顶时用 粗全局watershed标签/分块watershed+union-find传播 → 分块vectorize按全局ID dissolve(见记忆 changzhi-fullcity-run)。

**长治后续两测(2026-06-19~21,从保存 idmap.npy+cls_of.pkl,零重推理):** ① **Q1 巨斑分离+字段Chaikin(prefecture曲线可行性)**:去建筑类后 **字段 topojson+Chaikin FIT(不OOM,373726 polys)→ 实证建筑路网巨斑是 topojson 内存元凶**;但字段 smooth 慢(~7h:Chaikin 6h+resolve 3.7h),且字段↔建筑合并的 `resolve_overlaps` 在 445811 polys **卡死 ~19h 无产物**(同县级函数不可扩展墙,recon 在 1.77M 同样卡死)→ **端到端不可行 / 不合适**。**结论:prefecture 要曲线不划算**(折线 vs Chaikin 曲线在城市级缩放下差异近不可见;折线版 G2 3h 干净完整,定为交付)。真要曲线的正确做法 = 单 coverage 一次 topojson 但对巨斑做**拓扑保持的变容差预简化**(非 split+post-merge,避开 resolve_overlaps 墙),工程更大、ROI 仍低。② **PROV 省级合并原理(用长治全局 idmap 当案例)**:切块逐块 vectorize(块边切碎 3702 实例)+按全局 ID dissolve vs 直接全局 → **实例数 Δ=0、面积 Δ=10.15km²(0.04%)→ 分块-dissolve 无缝重建=全局,省级合并原理验证可行**。**诚实范围**:只验"合并原理",未验"idmap 太大装不下→粗全局标签/union-find 传播"那半(长治 idmap 装得下没触发)。

**长治求曲线·两条平滑路实测(2026-06-21,从存盘 idmap 零重推理):** ③ **A 高 tol 一次性平滑(tol=45,单 coverage 不拆):OOM 失败(EXIT 137)**。vectorize 445081 polys(52min)→ coverage_simplify(tol=45) 253s OK → 但**全市 topojson 仍 OOM**。⟹ **topojson 内存是弧数(路网路口)驱动、非顶点驱动 → 高 tol 减顶点救不了 → prefecture 一次性平滑确认不可行**(印证弧数墙)。④ **B 分块 per-tile Chaikin + 按全局 ID dissolve:成功(124min)**。坑+修:质心 sjoin 回贴 parcel_id → dissolve 归错组 → 类别打乱(林 6954→2375 是破绽);**修法 = parcel_id 当 class_id 喂 smooth 按位 carry(生产可靠)+ 平滑后类别取 cls_of[parcel_id](零 sjoin)**。修后 **294,393 地块(≈G2 293,788)、类别与 G2 几乎一致(林6854/耕3580/草1703/建1332/水288/园167/荒27)、slivers 仅 567(buggy 版 4483)、放大可见梯田曲线边、块边接缝基本愈合**;代价面积 13,941.7 vs G2 14,111.7(−1.2%,分块各自 simplify 在块边轻微侵蚀)。**总结论:prefecture+ 要曲线 → 分块 per-tile 平滑 + dissolve(每块县级口径 topojson 永不 OOM、可扩省级);一次性平滑因弧数墙不可行。** 脚本 run_tiled_smooth.py(B)/run_hitol_smooth.py(A);成品 changzhi_tiled_smooth.parquet(本地 sidecar/changzhi_product/)。

**⑤ 自动分流落地(smooth_dispatch.py,additive 不改 parcel_pipeline):** `smooth_auto(idmap, cls_of, tr4, crs, tol, iters, max_global_parcels=150000)` —— **地块数 ≤ 15万走全局 smooth_coverage(最干净/面积最准),> 15万自动转分块 smooth+dissolve(可扩省级,~−1%面积)**,返回同格式曲线 gdf(class_id+geometry)。阈值依据:榆中~10万 fit / 长治~29万 OOM,topojson 内存 ~∝ parcels³(路网连通),取保守 15万(黄土高原密度 ≈ ~5000km²),`max_global_parcels` 可调。烟雾测试(test_smooth_dispatch.py):同小窗口 全局 vs 分块 类别面积 Δ=0.04km²(逐类保真)。**future 区县 workflow 把 `pp.smooth_coverage(vectorize_idmap(...))` 换成 `smooth_dispatch.smooth_auto(idmap,cls_of,tr4,crs,...)` 即自动按规模出平滑产品。** **已接入生产入口(2026-06-21)**:`run_pipeline(..., smooth="coverage|auto|tiled")` + CLI `--smooth`(默认 coverage 向后兼容;auto=自动分流;tiled=强制分块;`_run_from_mosaic` 内懒加载 smooth_dispatch 避循环 import,del idmap 挪进分支)。验证:py_compile + import 无循环 + 签名 smooth 默认 coverage + CLI --help 通过;烟雾测试两路类别面积 Δ0.04km²。
