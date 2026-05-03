# Imagery Downloader — Tauri 重构设计

- **作者**：zhangfeng04@gmail.com（与 Claude Code 协作）
- **日期**：2026-05-03
- **状态**：草案，待用户审阅
- **替换对象**：`download_imagery.py`（Python CLI，1336 行）+ 三个辅助脚本

## 1. 目标与范围

### 1.1 要解决的问题
当前 `imagery_downloader/` 是一个 Python CLI 工具，输入是 GeoPackage（多边形集合），为每个多边形下载并裁剪卫星瓦片影像。痛点：

1. 输入受限于 GPKG 多边形，普通用户难以从"想下载某区域"直接到"准备好 GPKG"。
2. 进度只能靠另起 `monitor_download.py` 脚本轮询。
3. 失败后只能从头来过，没有部分重试。
4. Python + GDAL/rasterio 的环境配置对非开发用户门槛高。

### 1.2 重构后的目标
- **输入简化为"一个边界范围"**：地图框选 / 数值输入 / 矢量文件导入三种入口，全部归一为 `[minLon, minLat, maxLon, maxLat]`（WGS84）。
- **桌面应用**：Tauri 2.x，单一可执行文件，零运行时依赖。
- **纯 Rust 后端**：替换全部 Python 代码。
- **目标平台**：macOS（Apple Silicon + Intel）+ Windows x64。Linux 不在本期范围。

### 1.3 非目标（YAGNI）
- 不做断点续传持久化（崩溃后下载状态丢失，用户重新开始）。
- 不做多任务并发队列（UI 一次一个下载；后端 IPC 用 download_id 已为未来扩展铺路）。
- 不做用户登录 / 云同步。
- 本期不做 E2E 自动化测试。

## 2. 架构总览

```
┌─────────────────────── Tauri App (单一可执行文件) ──────────────────────┐
│                                                                          │
│   Frontend (WebView)                  Backend (Rust)                     │
│   ┌──────────────────────┐           ┌──────────────────────────┐        │
│   │ Map view (MapLibre)  │  invoke   │ commands/                │        │
│   │ Form panel           │ ────────► │ events ──────────────►   │        │
│   │ Progress panel       │           │ core/ (tiles, sources,   │        │
│   │ History panel        │ ◄──────── │   downloader, stitcher,  │        │
│   └──────────────────────┘           │   cog, vector, history)  │        │
│                                       └──────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.1 技术选型
| 层 | 选择 | 理由 |
|---|---|---|
| App 框架 | **Tauri 2.x** | 体积小、安全模型清晰、社区活跃 |
| 前端框架 | **Svelte 5 + Vite + TypeScript** | 编译产物最小、桌面工具不需要 React 生态 |
| 地图库 | **MapLibre GL JS** | 开源、免 token、raster XYZ 一行配置 |
| HTTP / 并发 | **`reqwest` + `tokio`** | 标准 Rust 异步栈 |
| 图像处理 | **`image` crate** | 纯 Rust，PNG/JPEG decode |
| 写 GeoTIFF | **`tiff` crate**（手写 COG）或 `cog-rs`（待评估） | 避免 GDAL 系统依赖 |
| 矢量解析 | **`geojson` + `shapefile` + `rusqlite`（解 GPKG WKB）** | 全 Rust，无 GDAL 依赖 |
| 状态持久化 | **本地 JSON 文件** | history.json，最多 10 条 |

### 2.2 Rust 模块拆分（每个模块单一职责，可独立单元测试）
| 模块 | 职责 | 主要依赖 | 不依赖 |
|---|---|---|---|
| `tiles` | 经纬度 ↔ 瓦片坐标、bbox→tile_range 数学 | 无 | 网络、文件 |
| `sources` | ESRI / Google / auto 的 URL 模板与测速 | `reqwest` | 写盘、UI |
| `downloader` | 并发下载、重试、取消、进度回调 | `reqwest`, `tokio`, `tokio_util` | 写盘、UI |
| `stitcher` | 多瓦片在内存里拼成一张 RGB | `image` | 网络、写盘 |
| `cog` | 写 COG GeoTIFF + 可选 preview.png | `tiff`, `image` | 网络 |
| `vector` | GeoJSON / Shapefile / GPKG → bbox + GeoJSON 几何 | `geojson`, `shapefile`, `rusqlite` | 网络、UI |
| `history` | 最近 10 次输入读写 | `serde_json` | 网络、UI |
| `commands` | Tauri command 入口，编排上面模块 | `tauri` | — |

## 3. IPC 契约（前后端通信）

### 3.1 Commands（请求-响应）
```ts
parse_vector_file(path: string)
  → { bbox: [number, number, number, number],
      geometry: GeoJSON,
      layer_count: number }
   | { error: "unsupported_format" | "no_geometry" | "io_error",
       message: string }

estimate_output(
  bbox: [number, number, number, number],
  zoom: number,
  source: "esri" | "google" | "auto"
) → { tile_count: number,
      pixel_w: number, pixel_h: number,
      est_size_mb: number, est_seconds: number }

start_download({
  bbox: [number, number, number, number],
  zoom: number,                    // 8..23
  source: "esri" | "google" | "auto",
  output_path: string,             // 用户选定的 .tif 文件路径
  max_concurrency: number,         // 默认 50
  retry_per_tile: number,          // 默认 3
  write_preview_png: boolean       // 默认 true
}) → { download_id: string }
   | { error: "invalid_bbox" | "output_not_writable" | ...,
       message: string }

cancel_download(download_id: string) → { ok: true }
retry_failed(download_id: string) → { ok: true } | { error, message }
list_history() → HistoryEntry[]
clear_history() → { ok: true }
```

### 3.2 Events（后端推送）
所有 events 都带 `download_id`，前端按 id 路由：

```ts
"download://progress"
  { download_id, completed, total, bytes_downloaded,
    current_speed_mbps, elapsed_sec, eta_sec }
  // 节流到约 4 Hz

"download://tile-failed"
  { download_id, x, y, z, attempt, error }

"download://stage"
  { download_id, stage: "downloading" | "stitching"
                       | "writing_cog" | "writing_preview" }

"download://done"
  { download_id, ok: true,
    output_path, preview_path?,
    bbox, zoom, source_used,
    duration_sec, total_tiles, failed_tiles, output_size_mb }
  | { download_id, ok: false, error: string }
```

### 3.3 设计要点
1. **download_id (UUID)**：未来支持队列模式零成本。
2. **estimate_output 单独抽出**：前端在用户调 zoom slider 时实时调用（防抖 200ms）。
3. **错误用判别联合**：前端可以针对错误类型做差异化恢复 UX。
4. **取消机制**：Rust 内部用 `tokio_util::sync::CancellationToken`，触发后所有 in-flight `reqwest` 立即 abort。

## 4. 数据流（典型用户路径）

### 4.1 路径 A：手动数值输入
1. 用户在 4 个输入框填经纬度 + 选 zoom + source。
2. 前端本地校验（minLon < maxLon、zoom 8..23）。
3. 防抖 200ms → invoke `estimate_output` → UI 显示瓦片数 / 像素 / 字节估算。
4. 用户选输出路径（Tauri dialog）→ 点"开始下载"。
5. invoke `start_download` → 返回 `download_id`。
6. Rust 侧：
   1. 注册 `CancellationToken`。
   2. `tiles::range_for_bbox(bbox, zoom) → Vec<TileCoord>`。
   3. emit `stage = "downloading"`。
   4. `tokio::stream` + `buffer_unordered(max_concurrency)`，每瓦片重试 N 次（200 ms / 800 ms / 3.2 s 指数退避）。
   5. 每 ~250 ms emit `progress`。
   6. emit `stage = "stitching"` → 拼成 RGBA Image（失败瓦片像素 alpha=0）。
   7. emit `stage = "writing_cog"` → 写 COG GeoTIFF（先写 `<output>.tmp`，成功后 rename）。
   8. 可选 emit `stage = "writing_preview"` → 写 preview.png。
   9. emit `done {ok:true}`。
7. 前端切到"完成"面板，写入 history（按 bbox+zoom+source 去重，更新时间戳上浮）。

### 4.2 路径 B：地图框选
- MapLibre 加载，默认底图 = 当前选中 source 的 raster XYZ。
- 切换 source → MapLibre raster source URL 替换（所见即所得预览）。
- 激活 rectangle draw → 拖拽矩形 → 算 bbox → 回填到数值输入框。
- 之后同路径 A。

### 4.3 路径 C：文件导入
- 拖文件入窗口 / 点"导入..."。
- invoke `parse_vector_file(path)`。
- Rust `vector` 模块按扩展名分发：
  - `.geojson` → `geojson` crate
  - `.shp` → `shapefile` crate（连带读 `.dbf`/`.shx`）
  - `.gpkg` → `rusqlite` 打开 + 读 `gpkg_geometry_columns` + 解 WKB
  - 多 layer 时取第一个非空 layer
- 返回 `{ bbox, geometry, layer_count }`。
- 前端在 MapLibre 上画几何（半透明填充）+ `fitBounds` + 回填 bbox。
- 之后同路径 A。

## 5. 错误处理

| 失败类别 | 处理方式 |
|---|---|
| 输入校验（bbox 颠倒、跨日期变更线、zoom 超界） | 前端阻断，红色 inline 提示 |
| estimate 超阈值（> 5 GB 或 > 50000 像素边） | 前端 warn 但不阻断 |
| 输出路径不可写 | start_download 立即返回 error，前端弹 toast + 文件选择器 |
| 单瓦片 HTTP 失败 | 后端自动重试 N 次（指数退避），最终失败计入 `failed_tiles`，emit `tile-failed` |
| 整体网络断开（连续 30 个瓦片连续失败） | 后端主动 cancel，emit `done {ok:false, error:"network_lost"}`；**保留已下载瓦片缓存**供 retry_failed 使用 |
| 磁盘写入失败 | emit `done {ok:false, error:"disk_full"/"io_error"}` |
| 用户取消 | CancellationToken 触发 → 所有 reqwest abort → emit `done {ok:false, error:"cancelled"}` → 删 .tmp 文件 |
| 进程崩溃 | Tauri panic hook 写本地 crash log；下载状态丢失 |

### 关键不变量
1. **不写半成品文件**：COG 先写 `<output>.tmp`，全部成功后 rename。取消或失败一律删 .tmp。
2. **失败瓦片不污染输出**：拼接时失败瓦片对应像素填透明（alpha 通道）。
3. **进度按瓦片数算，不按字节算**（字节因 JPEG 压缩波动大，瓦片更稳定）。
4. **session 内的 tile cache 保留到 retry_failed**：失败后只补缺的瓦片，不重下已成功的。

## 6. 功能取舍（vs 当前 Python 版本）

| 功能 | 当前 Python | Tauri 重构 | 说明 |
|---|---|---|---|
| 多源 ESRI / Bing / Google / auto | ✅ 全部 | **保留 ESRI + Google + auto，去掉 Bing** | Bing 需 quadkey 转换且画质/覆盖落后 |
| auto-zoom（按目标 m/px 自动调） | ✅ | **去掉** | bbox 输入下用户对精度有清晰预期，UI 实时显示像素/字节更直观 |
| 多边形输入 + 按多边形裁剪 | ✅ | **去掉**（核心简化） | 输入改为 bbox |
| 进度监控 | 单独脚本 | **UI 内置** | events 实时推送 |
| 取消 | Ctrl-C | **UI 取消按钮** | CancellationToken 实现 |
| 失败重试 | ❌（失败即记录） | **每瓦片自动重试 3 次 + UI "重试失败瓦片" 按钮** | 重大改进 |
| 历史记录 | ❌ | ✅ 最近 10 次 bbox + 参数 | 本地 JSON |
| 预览底图 | ❌ | ✅ 选定 source 的 XYZ 实时预览 | 几乎零成本 |
| 输出格式 | 普通 GeoTIFF | **COG GeoTIFF**（含金字塔） | GIS 软件加载更快 |
| 附加输出 | summary.json | `download_metadata.json` + 可选 `preview.png` | — |

## 7. 打包与分发

| 平台 | 产物 | 大小预估 | 签名 |
|---|---|---|---|
| macOS (Apple Silicon) | `.dmg` 含 `.app` | ~12 MB | Apple Developer ID 可选 |
| macOS (Intel) | `.dmg` 含 `.app` | ~12 MB | 同上 |
| Windows x64 | `.msi` (WiX) + `.exe` (NSIS) | ~10 MB | 不签也能跑（SmartScreen 警告） |

### 关键打包决策
1. **WebView 用系统自带**（macOS WKWebView / Windows WebView2）。
2. **WebView2 引导器嵌入安装包**（Windows 10 兼容，+~2 MB）。
3. **零 GDAL 依赖**：全 Rust crate 路线。
4. **CI 用 GitHub Actions** matrix `[macos-14, macos-13, windows-latest]` × `tauri-action`，tag push 自动发 release。

### 首次启动初始化
- macOS: `~/Library/Application Support/imagery-downloader/`
- Windows: `%APPDATA%\imagery-downloader\`
- 内含 `history.json` 与 `tile-cache/`（会话级，可定期清理）。

## 8. 测试策略

| 层 | 工具 | 范围 | 数量级 |
|---|---|---|---|
| Rust 纯函数单测 | `cargo test` | tiles 数学 / quadkey / bbox→tile_range / cog header bytes | ~30 |
| Rust 集成测试 | `cargo test --test` + `wiremock` | downloader 模拟各种 HTTP 响应 / vector 解析三格式 / stitcher→cog 写后反向校验 | ~15 |
| 前端组件测试 | `vitest` + `@testing-library/svelte` | 输入校验 / progress 显示 / history 交互 | ~10 |
| E2E | （暂不做） | — | 0 |

### 关键测试不变量
1. 给定相同 bbox + zoom + source，`tiles::range_for_bbox` 输出确定（property test）。
2. 写出的 COG 反读 → 像素维度匹配 + GeoTransform 投影回 bbox。
3. 取消后 `<output>.tmp` 与 `<output>.tif` 都不存在。
4. `failed_tiles` 数 = `total - 成功瓦片数`。
5. **网络层全部用 wiremock，不打真实 API**（CI 稳定性 + 离线可跑）。

## 9. 开放问题（待实现阶段决定）

1. **COG 写入选择**：`tiff` crate 手写 vs `cog-rs` 封装。后者社区使用度待评估。
2. **Apple Developer ID 签名**：是否申请、何时申请。不签的兜底是首次启动右键打开。
3. **Windows 代码签名**：暂不做，加文档提示用户对 SmartScreen 的处理。
4. **History 数据 schema**：精确字段集等实现时再敲。

## 10. 不在本设计内但应同期处理

1. 旧 `download_imagery.py` 等 Python 脚本：保留在仓库 `legacy/` 子目录或单独分支，避免新用户混淆。
2. 新仓库结构：`src-tauri/`（Rust）+ `src/`（前端）+ `docs/` + `legacy/`（旧 Python）。
3. 新 `README.md`：替换当前 Python CLI 文档。

---

## Plan C Implementation Status (2026-05-04)

✅ Frontend UI complete: 3-pane layout (input / map / progress+history).
✅ Three input modes: numeric form, map rectangle draw, file picker (parse pending Plan A).
✅ MapLibre raster preview with live source switching (esri/google), native rectangle drawing, fitBounds.
✅ Mock backend in `src-tauri/src/mocks/`: `start_download` simulates a
   5-sec download with 50 progress ticks, real CancellationToken-based
   cancel, real history persistence (atomic write).
✅ History panel with 10-newest-first cap, dedupe by bbox+zoom+source,
   click-to-restore.
✅ 16 vitest unit tests (validators, formatters, IPC wrapper, smoke).
✅ 6 cargo integration tests for history Store.
✅ The frontend is final-shape; Plan B replaces only `src-tauri/src/mocks/`.

Pending:
- Plan A: real `tiles`, `sources`, `downloader`, `stitcher`, `cog`, `vector` modules.
- Plan B: real Tauri commands wiring Plan A modules into `invoke_handler!`.
- Plan D Phases 2-4: CI workflow, release workflow, signing docs (deferred).

Known minor follow-ups:
- MapPanel attribution does not update on source change (snapshot at construct).
- MapPanel drop-target div has a Svelte a11y warning (no ARIA role).
- `chrono_iso_now()` returns `"epoch:N"` strings; Plan A will switch to ISO 8601 via `chrono`.
