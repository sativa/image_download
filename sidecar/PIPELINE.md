# PIPELINE.md — 1m 地块矢量管线(模型 → 无缝地物成品)

区域无关的端到端流程: **1m 影像 mosaic → 全局 watershed 推理 → idmap → 全县整体矢量化+平滑 →
裁界 → 标准后处理 → 无缝 7 类地物 GeoParquet**。从榆中(620123)专用链通用化而来,
不绑任何县/类别/降采样倍数。供部署、论文复现参考。

代码: `sidecar/parcel_pipeline.py`(通用管线) · `sidecar/postproc.py`(标准后处理) ·
`sidecar/yz_pipeline.py`(榆中示例 wrapper)。历史分阶段脚本 `yz_global_ffl.py` /
`yz_smooth2.py` / `yz_postproc.py` 保留作对照(算法已抽进通用模块)。

---

## ① 模型 — DINOv3-Sat 四头 BDDF

- **Backbone**: `dinov3-vitl16-sat493m`(DINOv3 ViT-L/16, 卫星 493M 预训练)。强跨域泛化是 1m 路线的核心卖点。
  权重位置: Mac `~/D/...`、.250 `/home/ps/landform/dinov3/dinov3-vitl16-sat493m`。
- **解码器**: `DinoV3FreqUNetBDDF`(`train_dino_1m_v3.py`)。DINOv3 低分辨率 patch 网格 + 输入多分辨率高分支,
  用 **FreqFusion**(频域感知上采样, `ff1/ff2/ff3`: /16→/8→/4→/2)逐级融合上采样 → UNet 风格密集预测。
- **四头(BDDF)**: forward 返回 `(cls, bnd, dist, frame_field)`
  - `cls` — 9 类 logits(softmax)。部署 7 类(设施大棚 8→建筑 6 merge; 类 0 背景)。
  - `bnd` — 边界概率(sigmoid)。
  - `dist` — 地块中心距离场(sigmoid); 峰值=地块中心 → watershed 种子。
  - `frame_field` — 帧场 `[B,4,H,W]`(c0,c2 复系数, Tanh)。**通用版默认不用**(见 §⑤ 被否方向)。
- **输入**: 11 通道 = 6(Esri RGB + Google RGB, 推理时 Google 缺则复制 Esri)+ 5(NDVI, 1m 路线零填充)。
  训练/推理一致用 `enhance6`(CLAHE+unsharp 锐化梯田/边缘对比)后 `norm6` 归一化。可选 GDLX 辅助头(梯田/坡地)。
- **权重**: 四头 ckpt 例 `/mnt/sda/zf/landform/results/dino_v3_bddf/last.pt`(或 `dino_v3_bddf_enh/best.pt`)。

---

## ② 端到端流程

1. **下载**: Rust CLI `imagery-downloader batch --regions <json> --source esri|google|auto --zoom 17`
   → 每 cell 单文件 YCbCr JPEG-GeoTIFF(EPSG:3857)。
2. **拼接 mosaic**: `rasterio.merge` 把 cell 拼成单张大图(全县或块级)。单图推理 → 无 cell 内缝。
3. **全局 /N watershed 推理**(`parcel_pipeline.infer_global`, 从 `yz_global_ffl` 抽出去 FFL):
   - 多 GPU **行带并行**(spawn, 每带绑一张卡): 448px 窗 / 224 stride(50% 重叠)滑窗前向。
   - 每窗 `softmax(cls)` / `sigmoid(dist)` / `sigmoid(bnd)` → **/N area-resize** 降到 /N 网格 →
     **Hann 窗加权累加**到全局 /N 累加器(`acc_cls/acc_dist/acc_bnd/cnt`)。
   - 主进程归并各带 → `cls = acc/cnt`。**一张全局 /N 概率图**, 无块边、无 cell 缝。
4. **build_idmap(ridge watershed)**(`dino_parcel_export.build_idmap`, 复用):
   - 耕地/园地(类 1/2): `dist` 峰值=种子 → `watershed(elevation=max(bnd, 1-dist))` ridge flood,
     边界头加固边缘 → 实例; 每实例按 cls 概率定耕地/园地。
   - 其余类(3 林/4 草/5 水/6 建/7 荒): argmax 连通域 → 实例。
   - 空白像素 EDT 最近邻补齐 → **全覆盖无空洞 idmap**(`int32`)+ `cls_of`(实例→类)。
5. **全县整体矢量化 + Chaikin 平滑**(`parcel_pipeline.vectorize_idmap` + `smooth_coverage`, 从 `yz_smooth2` 抽出):
   - `rasterio.features.shapes` 出干净分区多边形(无缝 coverage)。
   - `coverage_simplify(tol)` **全县整体** 去 /N 像素阶梯(共享边精确一致, 顶点降 ~2x)。
   - **全县一次** `topojson.Topology(shared_coords=False)` → 共享边只存一份 arc。
   - 每 arc `chaikin_arc(iters)` 角点切割平滑(**端点=节点固定**)→ 共享边两侧逐点一致 → **零重叠无缝**。
6. **裁县界**(可选, `clip_to_boundary`): 给了 boundary(parquet/geojson)就真几何 `intersection` 裁;
   `covered_by` 快筛全内的, 仅边界相交块做裁切。否则跳过。
7. **标准后处理**(`postproc.run_postproc`, 见 §③)→ 无缝标准成品。
8. **产品**: EPSG:4326 GeoParquet, 列 `gid, class_id, label, label_en, rgb_hex, area_m2, geometry`。

---

## ③ 标准后处理 — `postproc.py`

**所有模型矢量成品的统一收尾**(region-agnostic), 顺序由 `run_postproc` 串起:

```
fix_invalid → eliminate_slivers → fill_gaps_holes → fix_invalid → standardize
```

全部在米制 CRS(默认 UTM `EPSG:32648`)下做几何运算, 最后 `standardize` 转回 EPSG:4326。

| 函数 | 作用 | 判据 / 阈值含义 |
|---|---|---|
| `fix_invalid` | make_valid 修自相交/退化 | `is_valid==False` → `make_valid`; explode 到单 Polygon, 丢线/点碎屑。平滑/裁切后必跑的几何卫生。 |
| `eliminate_slivers` | 细长线状碎屑**并入**最长共享边邻块(非删, 保无缝) | sliver 判据(`is_sliver`): ① `w=area/peri < w_min(2m)` 且 `PP=4πA/P² < 0.30`(细**且**长条, 排除小方块 PP≈0.785); ② `PP < pp_min(0.05)` 且 `area < a_max`(极不紧凑细线)。并入策略: 取**共享边界最长**邻块属性 union; 无邻保留不删; 迭代收敛。 |
| `fill_gaps_holes` | 修补"空白图斑": 图斑间空隙 + 县界内未覆盖 + 空心洞 | ① `boundary.difference(union)`(给了 boundary)= 县界内未覆盖区; ② union 后内部 interior ring = 空心洞。每块空白按**多数票/最长共享边**邻块类别合并填补 → 无缝。`min_gap_area` 过滤碎屑。 |
| `standardize` | 字段标准化 + 统一 EPSG:4326 | 加 `gid` / `area_m2`(UTM 米制下算)/ `label` / `label_en` / `rgb_hex`(从 classes schema 查)。classes 接受 dict 或 list-of-tuple(同 `dino_parcel_export.CLASSES`)。 |

`find_slivers` / `is_sliver` 可单独用于出"清理前后对比图"。

---

## ④ 怎么在新区域跑

**命令行**(任意区域, 不改代码):
```bash
python parcel_pipeline.py \
  --mosaic /path/to/region_mosaic.tif \
  --weights /path/to/dino_v3_bddf.pt \
  --backbone /path/to/dinov3-vitl16-sat493m \
  --boundary /path/to/county_boundary.parquet   # 或 none 跳过裁界 \
  --out /path/to/region_parcels.parquet \
  --downscale 4 --smooth-iters 2 --tol 5 \
  --classes classes.json    # 可选; 空=默认7类 \
  --gpus 0,1,2,3 --utm EPSG:32648
```
- `--classes` JSON 格式: `[[id, "中文", "english", [r,g,b]], ...]`(留空=默认 7 类)。
- `--utm` 改成目标区域所在 UTM 带(榆中 48N=32648; 西藏 91E→46N=32646)。
- `--downscale` 越大越快/越糙(大 mosaic 一遍过); `--smooth-iters` 控制曲线顺滑度(0=关)。

**作为库调用**:
```python
import parcel_pipeline as pp, postproc
gdf, report = pp.run_pipeline(mosaic, weights, backbone, out,
                              boundary="county.parquet", downscale=4, smooth_iters=2,
                              classes=None, gpus=["0","1","2","3"], utm="EPSG:32648")
# 或对任意已有矢量(任意 backend)只跑标准收尾:
clean, rep = postproc.run_postproc(my_gdf, classes, boundary=bnd_geom, utm="EPSG:32648")
```

**榆中示例**(thin wrapper, 仅传榆中常量): `python yz_pipeline.py --gpus 0,1,2,3`。

---

## ⑤ 关键设计决策 & 被否方向

- **FFL 帧场 → 改 dist/bnd ridge watershed + 拓扑保持**: 帧场正则(Frame Field Learning)矢量化使多边形
  **过直**(直线化过度), 且**逐实例多边形重叠**(各实例独立拟合, 不共享边)。通用版默认 **不用 FFL**,
  改用 `dist` 峰值种子 + `max(bnd, 1-dist)` ridge watershed 出实例, 再用 **topojson 共享 arc + Chaikin**
  保拓扑 → 无缝。(四头 ckpt 的 frame_field 输出在推理时忽略 `o[3]`。)
- **全局累加器 → 而非 per-cell / 分块**: per-cell 推理慢且有 cell 缝; 分块(block)在切线处留**白缝/重叠**。
  改 **全局一张 /N 网格 Hann 累加** → 一次 watershed → 整县无缝 partition(`yz_full_pipeline` 的块级方案
  靠块内单图消内缝 + clamp 治块间白缝, 全局累加器从根上免缝)。
- **全县整体 topojson + Chaikin arc → 而非逐多边形平滑**: 逐多边形独立 Chaikin 会让共享边两侧**各动各的**
  → 局部重叠/缝。全县一次 topojson 后**共享边只一份 arc**, 平滑端点固定 → 两侧逐点一致 → 严格无缝。
  `shared_coords=False`(junction/coord-hash)比 `shared_coords=True` 在 coverage_simplify 输出上合并共享边更可靠
  (零 Chaikin 重叠, 更少 arc), 故通用版用之, 平滑后无需额外 snap。
- **巨型图斑边界跳过 Chaikin(治"悬空线段"伪影)**: 连通的**建筑/道路网**被连通域标号成一个巨型多边形(榆中 510万顶点/上千洞)。
  Chaikin 对这种超复杂路网边界逐弧平滑 → **节点处狂产细楔形 sliver(w<1m)= 悬空线段**。修法 `smooth_coverage._giant_adjacent_arcs`:
  至少一侧邻接巨型图斑(顶点≥5万)的 arc **跳过 Chaikin、保 coverage_simplify 直边**(道路本就直边), 田块间 arc 照常 Chaikin
  保曲线 → 路网边界不再生楔形(榆中建筑 maxV 510万→69万, 楔形 sliver→0)。per-arc 选择性平滑, 默认开。
- **"伪影 vs 真地物"诊断纪律(别盲目删 sliver)**: 细 sliver 两类 —— ① **Chaikin 节点楔形 = 真伪影**(上条已治);
  ② **真实细线状地物**(窄梯田条/小道/田埂)= 1m 忠实捕捉、**该保留**。榆中残留 w<0.5 sliver 65%是耕地(真窄梯田)、仅6%建筑(路),
  叠影像验证落在真梯田结构上。**只能"同类合并"且仅 -8%**(底层就是细的、多无同类邻居)→ **跨类并=毁真地物, 禁止**。
  结论: 真伪影清掉后剩的细线是真地物不是错误; 嫌出图碎用**显示层 MMU(按面积隐藏极小)**, 不改数据。
- **postproc 治伪影**: `drop_tiny_holes`(删近零面积退化 interior ring=节点微洞, 留嵌套/大洞)+ `run_postproc(skip_gaps=)`
  (无缝 coverage 输入跳过 fill_gaps 全量 union, 防误填路网合法内部洞致面积虚高 +90km²)+ STRtree 逐对消重叠(避全量 unary_union)。
- **榆中终版 = `yuzhong_FINAL.parquet`**(= SMOOTH3_v2, giant-skip 后): 74177 地块, 楔形伪影清零, 面积 3166.2(-0.006%), valid, 零重叠,
  田块曲线保留; 旧 `yuzhong_SMOOTH3_chaikin3`(含 510万顶点建筑怪物)已被取代。
- **DLTB(三调)= 权威真值**, 面积重建对比(`area_recon`, 榆中口径)校核, 总面积域内 ±0.3%。
- **评估口径**(见 `cropland-seg-honest-eval`): 1m 地块级、跨县 + 跨省两口径; 跨省必报 argmax(部署点)和 best-thr(上界)。

---

## 文件索引

| 文件 | 角色 |
|---|---|
| `parcel_pipeline.py` | **通用管线**: `infer_global` / `idmap_from_heads` / `vectorize_idmap` / `smooth_coverage` / `clip_to_boundary` / `run_pipeline` |
| `postproc.py` | **标准后处理**: `fix_invalid` / `eliminate_slivers` / `fill_gaps_holes` / `standardize` / `run_postproc` |
| `yz_pipeline.py` | 榆中示例 wrapper(传 620123 常量) |
| `dino_parcel_export.py` | `build_idmap`(ridge watershed + 连通域) + 7 类 CLASSES/NAME/HEX schema |
| `dino_parcel_eval.py` | `infer_heads`(Hann 滑窗) / `dist_peak_instances`(对照单 cell 推理) |
| `yz_global_ffl.py` / `yz_smooth2.py` / `yz_postproc.py` | 历史分阶段脚本(算法已通用化; `yz_postproc` 现 re-export `postproc`) |
