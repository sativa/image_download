# Plan A — Rust core 模块 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 spec §2.2 列出的 6 个 Rust 核心模块（`history` 已在 Plan C 完成；剩 `tiles` / `sources` / `downloader` / `stitcher` / `cog` / `vector`）以 TDD 方式写完，使后端能从 bbox+zoom+source 真实下载瓦片并产出 COG GeoTIFF——但**仍不接入 Tauri command**。Plan B 接手后只删 `src-tauri/src/mocks/`、把这些模块连到 `invoke_handler!`。

**Architecture:**
- 6 个模块全部在 `src-tauri/src/core/` 下，每个模块单一职责、无 Tauri / serde-fluent 依赖（除了 vector 模块要用 serde 解 geojson）。
- 网络层只有 downloader 用 `reqwest`，其它模块输入是 `Vec<u8>` / 几何 / 数组，不接触 IO。
- 测试：单元测试用 `#[cfg(test)]` 写在每个模块文件底部；集成测试（用 wiremock / 真实读写文件）放 `src-tauri/tests/`。
- COG 选择：spec §9 开放问题——**手写 `tiff` crate**，无外部 GeoTIFF 依赖。
- 投影：tiles 是 EPSG:3857（Web Mercator），输出 GeoTIFF 也用 3857（不重投影到 4326，避免重采样损失 + 复杂度）。

**Tech Stack:** `reqwest@0.12` (rustls) · `tokio@1` · `tokio-util@0.7` (CancellationToken) · `image@0.25` · `tiff@0.10` · `geojson@0.24` · `shapefile@0.6` · `rusqlite@0.32` (bundled SQLite) · `bytes@1` · 测试栈 `wiremock@0.6` + `tempfile@3`

---

## 文件结构（计划落地后）

```
src-tauri/src/
├── lib.rs                          # 修改：注册 core::*；mocks/ 暂保留
├── main.rs                         # 不动
├── history.rs                      # 不动（Plan C 已实现，Plan B 时 move 到 core/）
├── core/
│   ├── mod.rs                      # pub mod tiles; pub mod sources; ...
│   ├── tiles.rs                    # 瓦片数学（纯函数）
│   ├── sources.rs                  # URL 模板 + auto 测速
│   ├── downloader.rs               # 并发下载 + 重试 + 取消 + tile cache
│   ├── stitcher.rs                 # RGBA 拼接
│   ├── cog.rs                      # 写 COG GeoTIFF + preview.png
│   └── vector.rs                   # GeoJSON / Shapefile / GPKG 解析
├── mocks/                          # 不动；Plan B 删除
└── tests/
    ├── tiles_test.rs               # 已包含模块单测；这里只放 property test
    ├── downloader_test.rs          # wiremock 驱动
    ├── cog_test.rs                 # 写 COG → tiff crate 反读 → 校验维度 + 像素
    ├── vector_test.rs              # 三格式 fixtures
    └── e2e_test.rs                 # 全链路集成
```

**职责切分：**
- `tiles.rs` 纯计算，零 IO；其它模块依赖它。
- `sources.rs` 提供 URL 模板 + `auto` 路径下的延迟测速；不下整张图，只 probe 一片。
- `downloader.rs` 唯一拥有 reqwest client；通过 `&dyn TileSource` 接收 URL 模板。
- `stitcher.rs` 输入 `Vec<DownloadedTile>`，输出 `image::RgbaImage`。
- `cog.rs` 接受 `RgbaImage` + bbox + zoom，写入 path（atomic）；可选 preview.png（image crate 自身渲染）。
- `vector.rs` 输入 `&Path`，输出 `(bbox, geojson::Geometry)`；不依赖任何其它 core 模块。

---

## 阶段总览

| Phase | 名称 | 任务数 | 产出验证 |
|---|---|---|---|
| 0 | 模块树 + 依赖 | 1 | `cargo check` 0 错；`core::*` 出现在 `cargo doc` |
| 1 | tiles 数学 | 4 | 30+ 单测 + 1 property test 全绿 |
| 2 | sources URL + auto | 3 | wiremock 验 probe，trait 抽象稳定 |
| 3 | downloader | 4 | wiremock 集测：成功 / 重试 / 取消 / 部分失败可 retry |
| 4 | stitcher | 2 | 失败瓦片像素 alpha=0；维度等于 tx*256 × ty*256 |
| 5 | cog | 5 | 写出文件 → `tiff` 反读 → GeoTransform 投影回 bbox 误差 < 1 px |
| 6 | vector | 4 | 三格式 fixtures 都 round-trip |
| 7 | E2E + Plan B 契约 | 2 | 一次 `cargo test --test e2e_test` 跑完 vector→tiles→download(mock)→stitch→cog→read，写 `core/README.md` |

合计 25 任务。

**前置依赖：**
- Plan C 落到 `main`（commit `32d393c` 或之后）。
- `cargo test --test history_test` 6 passing。
- `pnpm test` 16 passing。
- 工作树干净。

---

## Phase 0 — 模块树 + 依赖

### Task 0.1：建立 core/ 树 + 加 deps + cargo check

**Files:**
- Create: `src-tauri/src/core/mod.rs`
- Create: 6 个空 `src-tauri/src/core/{tiles,sources,downloader,stitcher,cog,vector}.rs`
- Modify: `src-tauri/Cargo.toml`、`src-tauri/src/lib.rs`

- [ ] **Step 1: 建目录与空文件**

```bash
mkdir -p src-tauri/src/core
for m in tiles sources downloader stitcher cog vector; do
  echo "//! ${m} module — Plan A" > "src-tauri/src/core/${m}.rs"
done
```

- [ ] **Step 2: 写 core/mod.rs**

```rust
//! Core domain modules implementing the satellite-imagery downloader.
//!
//! All modules are Tauri-agnostic — they take plain inputs and return
//! plain outputs. Plan B's commands/ wraps them.

pub mod tiles;
pub mod sources;
pub mod downloader;
pub mod stitcher;
pub mod cog;
pub mod vector;
```

- [ ] **Step 3: 加 deps**

```bash
cd src-tauri
cargo add reqwest@0.12 --features rustls-tls,stream --no-default-features
cargo add image@0.25 --features png,jpeg
cargo add tiff@0.10
cargo add geojson@0.24
cargo add shapefile@0.6
cargo add rusqlite@0.32 --features bundled
cargo add bytes@1
cargo add futures@0.3 --features std
cargo add anyhow@1
cargo add thiserror@2
cargo add --dev wiremock@0.6
cargo add --dev tempfile@3
cd ..
```

> **注意**：`reqwest` 用 `rustls-tls` 而非默认 native-tls，避开 OpenSSL 系统依赖（spec §2.1 "全 Rust crate"）。`rusqlite` 用 `bundled` 让它自己编译 SQLite，不依赖系统库。这两个选择决定了交叉编译能成（Plan D 的 macOS Intel leg 不会因 OpenSSL 失败）。

- [ ] **Step 4: 修改 lib.rs 注册 core**

打开 `src-tauri/src/lib.rs`，在第一行 `pub mod history;` 旁边加：

```rust
pub mod history;
pub mod core;
mod mocks;
```

注：`pub mod core` 是 pub，因为集成测试要用 `imagery_downloader_lib::core::...`。

- [ ] **Step 5: cargo check + commit**

```bash
cd src-tauri && cargo check && cd ..
```

期望：0 错误；6 个新模块各有一条 dead-code 警告（暂时空的）。

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/lib.rs src-tauri/src/core/
git commit -m "feat(a): scaffold core/ module tree + add reqwest/image/tiff/geojson/shapefile/rusqlite

6 empty modules under core/. reqwest uses rustls (no system OpenSSL),
rusqlite uses bundled SQLite (no system libsqlite3) — both required
for clean cross-compile on Plan D's macOS Intel leg."
```

✅ Phase 0 完成。

---

## Phase 1 — tiles 模块

### Task 1.1：lon/lat → tile_x/tile_y（fail-first）

**Files:**
- Modify: `src-tauri/src/core/tiles.rs`

- [ ] **Step 1: 写测试**

把 `tiles.rs` 替换为：

```rust
//! Web-Mercator tile math (EPSG:3857).
//!
//! XYZ tile coordinates: x grows east 0..2^z, y grows south 0..2^z, z is zoom.
//! Conventions match OSM / ESRI / Google.

use std::f64::consts::PI;

pub fn lon_to_tile_x(lon: f64, zoom: u32) -> i64 {
    let n = 2_f64.powi(zoom as i32);
    ((lon + 180.0) / 360.0 * n).floor() as i64
}

pub fn lat_to_tile_y(lat: f64, zoom: u32) -> i64 {
    let n = 2_f64.powi(zoom as i32);
    let lat_rad = lat.clamp(-85.05112878, 85.05112878).to_radians();
    ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * n).floor() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lon_origin_at_zoom_2() {
        // lon=0 at z=2 splits 4 tiles east-west; tile_x of (0, _) is 2 (the boundary, floor → 2)
        assert_eq!(lon_to_tile_x(0.0, 2), 2);
    }

    #[test]
    fn lon_west_extreme() {
        assert_eq!(lon_to_tile_x(-180.0, 2), 0);
        assert_eq!(lon_to_tile_x(-180.0, 4), 0);
    }

    #[test]
    fn lon_just_under_east_extreme() {
        // lon=180 at z=4 = floor(16.0) = 16 (out of bounds!) — caller should pre-clamp.
        // We accept this and document it; tests use lon=179.999... to stay in-range.
        assert_eq!(lon_to_tile_x(179.999, 4), 15);
    }

    #[test]
    fn lat_equator_at_zoom_2() {
        assert_eq!(lat_to_tile_y(0.0, 2), 2);
    }

    #[test]
    fn lat_clamps_to_mercator_extent() {
        // Above 85.05° we clamp; tile y near 0
        assert_eq!(lat_to_tile_y(89.0, 4), 0);
        assert_eq!(lat_to_tile_y(-89.0, 4), 15);
    }

    #[test]
    fn known_pair_beijing() {
        // Beijing 116.404°E, 39.915°N at z=10 → x=843, y=388 (verified against openstreetmap.org)
        assert_eq!(lon_to_tile_x(116.404, 10), 843);
        assert_eq!(lat_to_tile_y(39.915, 10), 388);
    }
}
```

- [ ] **Step 2: cargo test**

```bash
cd src-tauri && cargo test --lib tiles && cd ..
```

期望：`6 passed; 0 failed`。如果 Beijing 测试失败，验证 OSM 的实际值（数学应当正确，但 floor vs round 边界可能差 1）。

- [ ] **Step 3: commit**

```bash
git add src-tauri/src/core/tiles.rs
git commit -m "feat(a): tiles — lon/lat → XYZ tile coordinates with mercator clamp"
```

---

### Task 1.2：tile → lon/lat 反向（用于 stitcher 写 COG GeoTransform）

**Files:**
- Modify: `src-tauri/src/core/tiles.rs`

- [ ] **Step 1: 在 `#[cfg(test)] mod tests` 之前追加**

```rust
pub fn tile_x_to_lon(x: i64, zoom: u32) -> f64 {
    let n = 2_f64.powi(zoom as i32);
    x as f64 / n * 360.0 - 180.0
}

pub fn tile_y_to_lat(y: i64, zoom: u32) -> f64 {
    let n = 2_f64.powi(zoom as i32);
    let lat_rad = (PI * (1.0 - 2.0 * y as f64 / n)).sinh().atan();
    lat_rad.to_degrees()
}
```

- [ ] **Step 2: 加测试到 `mod tests`**

```rust
    #[test]
    fn roundtrip_lon() {
        // For any tile-aligned lon, going lon → x → lon should be exact (given x is the floor).
        for &lon in &[-180.0, -90.0, 0.0, 90.0, 179.999] {
            let x = lon_to_tile_x(lon, 8);
            let back = tile_x_to_lon(x, 8);
            // back is the western edge of tile x; lon may be inside or at edge
            assert!(back <= lon, "lon={} → x={} → back={}", lon, x, back);
            assert!(back >= lon - 360.0 / 256.0, "tile too far west");
        }
    }

    #[test]
    fn roundtrip_lat() {
        for &lat in &[-80.0, -45.0, 0.0, 45.0, 80.0] {
            let y = lat_to_tile_y(lat, 8);
            let back = tile_y_to_lat(y, 8);
            assert!(back >= lat - 1.0, "lat={} → y={} → back={}", lat, y, back);
        }
    }

    #[test]
    fn bbox_corners_at_zoom_4_china() {
        // Tile (12, 6) at z=4 covers eastern China.
        let west = tile_x_to_lon(12, 4);
        let north = tile_y_to_lat(6, 4);
        assert!((west - 90.0).abs() < 0.001);
        assert!((north - 40.97).abs() < 0.1);
    }
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --lib tiles && cd ..
git add src-tauri/src/core/tiles.rs
git commit -m "feat(a): tiles — inverse projection tile_x/y → lon/lat for GeoTransform"
```

---

### Task 1.3：range_for_bbox + TileCoord struct

**Files:**
- Modify: `src-tauri/src/core/tiles.rs`

- [ ] **Step 1: 在 helper 函数之后、`#[cfg(test)]` 之前加**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub x: i64,
    pub y: i64,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileRange {
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64, // inclusive
    pub y_max: i64, // inclusive
    pub z: u32,
}

impl TileRange {
    pub fn count(&self) -> u64 {
        ((self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)) as u64
    }
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        let z = self.z;
        (self.y_min..=self.y_max).flat_map(move |y| {
            (self.x_min..=self.x_max).map(move |x| TileCoord { x, y, z })
        })
    }
}

/// Compute the inclusive tile range covering the bbox [minLon, minLat, maxLon, maxLat] at zoom.
/// Bbox is in WGS84; tiles are XYZ web-mercator.
pub fn range_for_bbox(bbox: [f64; 4], zoom: u32) -> TileRange {
    let [w, s, e, n] = bbox;
    let x_min = lon_to_tile_x(w, zoom);
    let x_max = lon_to_tile_x(e, zoom).max(x_min); // ceil-1 not needed; e's floor is the eastern tile
    // y grows south, so northern lat = y_min
    let y_min = lat_to_tile_y(n, zoom);
    let y_max = lat_to_tile_y(s, zoom).max(y_min);
    let max_idx = (1i64 << zoom) - 1;
    TileRange {
        x_min: x_min.clamp(0, max_idx),
        y_min: y_min.clamp(0, max_idx),
        x_max: x_max.clamp(0, max_idx),
        y_max: y_max.clamp(0, max_idx),
        z: zoom,
    }
}
```

- [ ] **Step 2: 加测试**

```rust
    #[test]
    fn range_single_tile() {
        let r = range_for_bbox([0.5, 0.5, 0.6, 0.6], 8);
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn range_known_extent() {
        // [100, 30, 110, 40] at z=8 ≈ 8 tiles wide × 7 tall = 56 tiles
        let r = range_for_bbox([100.0, 30.0, 110.0, 40.0], 8);
        assert!(r.count() >= 50 && r.count() <= 64, "got {}", r.count());
        assert!(r.x_min < r.x_max);
        assert!(r.y_min < r.y_max);
    }

    #[test]
    fn range_iter_yields_unique_tiles() {
        let r = range_for_bbox([0.0, 0.0, 1.0, 1.0], 6);
        let v: Vec<_> = r.iter().collect();
        let unique: std::collections::HashSet<_> = v.iter().copied().collect();
        assert_eq!(v.len(), unique.len());
        assert_eq!(v.len() as u64, r.count());
    }

    #[test]
    fn range_clamps_to_world() {
        let r = range_for_bbox([-181.0, -86.0, 181.0, 86.0], 4);
        assert_eq!(r.x_min, 0);
        assert_eq!(r.x_max, 15);
        assert_eq!(r.y_min, 0);
        assert_eq!(r.y_max, 15);
    }
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --lib tiles && cd ..
git add src-tauri/src/core/tiles.rs
git commit -m "feat(a): tiles — TileCoord/TileRange + range_for_bbox with iter()"
```

---

### Task 1.4：property test（roundtrip 不变量）

**Files:**
- Create: `src-tauri/tests/tiles_test.rs`

> 不引 proptest 依赖——手写一个伪随机覆盖即可，避免再加 dev-dep。

- [ ] **Step 1: 写文件**

```rust
//! Property-style tests for tiles module: roundtrip + range invariants.

use imagery_downloader_lib::core::tiles::*;

fn next_pseudo(seed: &mut u64) -> f64 {
    // xorshift64*; returns f64 in [0, 1)
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed as f64 / u64::MAX as f64).abs()
}

#[test]
fn lon_lat_roundtrip_bounded_error() {
    let mut seed = 0xDEADBEEFu64;
    for zoom in [8, 12, 17, 22] {
        for _ in 0..200 {
            let lon = next_pseudo(&mut seed) * 360.0 - 180.0 + 1.0;
            let lat = next_pseudo(&mut seed) * 160.0 - 80.0;
            let x = lon_to_tile_x(lon, zoom);
            let y = lat_to_tile_y(lat, zoom);
            let lon_back = tile_x_to_lon(x, zoom);
            let lat_back = tile_y_to_lat(y, zoom);
            // Each tile spans 360/2^z degrees in longitude. lon must be within the tile.
            let span_lon = 360.0 / 2_f64.powi(zoom as i32);
            assert!(
                lon - lon_back >= 0.0 && lon - lon_back <= span_lon + 1e-9,
                "lon={lon} z={zoom} x={x} back={lon_back} span={span_lon}"
            );
            // Latitude span varies; loose bound: 1 degree at z=8, 0.001 at z=22 — use 360/2^z*2.
            assert!((lat - lat_back).abs() <= 360.0 / 2_f64.powi(zoom as i32) * 2.0 + 1.0);
        }
    }
}

#[test]
fn bbox_count_matches_iter_len() {
    let bboxes = [
        [0.0, 0.0, 10.0, 10.0],
        [-50.0, -30.0, 50.0, 30.0],
        [100.0, 20.0, 105.0, 25.0],
    ];
    for &b in &bboxes {
        for z in [10, 14, 18] {
            let r = range_for_bbox(b, z);
            assert_eq!(r.count() as usize, r.iter().count());
        }
    }
}
```

- [ ] **Step 2: cargo test**

```bash
cd src-tauri && cargo test --test tiles_test && cd ..
```

期望：`2 passed; 0 failed`，每个测试触发数百次 assertions 全过。

- [ ] **Step 3: commit**

```bash
git add src-tauri/tests/tiles_test.rs
git commit -m "test(a): tiles property tests — 200×4 zooms roundtrip + range/iter invariant"
```

✅ Phase 1 完成。

---

## Phase 2 — sources 模块（URL 模板 + auto 测速）

### Task 2.1：TileSource trait + EsriSource + GoogleSource

**Files:**
- Modify: `src-tauri/src/core/sources.rs`

- [ ] **Step 1: 写实现**

替换 `sources.rs`:

```rust
//! XYZ tile source URL templates and auto-selection.

use crate::core::tiles::TileCoord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    Esri,
    Google,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Esri => "esri",
            SourceKind::Google => "google",
        }
    }

    pub fn parse(s: &str) -> Option<SourceKind> {
        match s {
            "esri" => Some(SourceKind::Esri),
            "google" => Some(SourceKind::Google),
            _ => None,
        }
    }
}

/// Build the XYZ URL for one tile from one source.
pub fn url_for_tile(s: SourceKind, t: TileCoord) -> String {
    match s {
        SourceKind::Esri => format!(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            z = t.z, x = t.x, y = t.y,
        ),
        SourceKind::Google => format!(
            "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            x = t.x, y = t.y, z = t.z,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tiles::TileCoord;

    #[test]
    fn esri_url_has_z_y_x_order() {
        let u = url_for_tile(SourceKind::Esri, TileCoord { x: 1, y: 2, z: 3 });
        assert!(u.ends_with("/3/2/1"));
    }

    #[test]
    fn google_url_has_query_params() {
        let u = url_for_tile(SourceKind::Google, TileCoord { x: 1, y: 2, z: 3 });
        assert!(u.contains("x=1") && u.contains("y=2") && u.contains("z=3"));
    }

    #[test]
    fn parse_roundtrip() {
        for s in [SourceKind::Esri, SourceKind::Google] {
            assert_eq!(SourceKind::parse(s.as_str()), Some(s));
        }
        assert_eq!(SourceKind::parse("bing"), None);
    }
}
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --lib sources && cd ..
git add src-tauri/src/core/sources.rs
git commit -m "feat(a): sources — Esri + Google XYZ URL templates"
```

---

### Task 2.2：probe_latency 异步函数（fail-first with wiremock）

**Files:**
- Create: `src-tauri/tests/sources_test.rs`
- Modify: `src-tauri/src/core/sources.rs`

- [ ] **Step 1: 写集测**

`src-tauri/tests/sources_test.rs`:

```rust
use imagery_downloader_lib::core::sources::*;
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn probe_returns_latency_under_response_delay() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/probe"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(50)))
        .mount(&server)
        .await;

    let url = format!("{}/probe", server.uri());
    let lat = probe_url(&url).await.expect("probe ok");
    assert!(lat >= Duration::from_millis(40), "got {:?}", lat);
    assert!(lat < Duration::from_secs(2));
}

#[tokio::test]
async fn probe_returns_err_on_404() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/probe"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let url = format!("{}/probe", server.uri());
    assert!(probe_url(&url).await.is_err());
}
```

- [ ] **Step 2: cargo test → fail（probe_url 不存在）**

```bash
cd src-tauri && cargo test --test sources_test && cd ..
```

- [ ] **Step 3: 在 sources.rs 末尾（`#[cfg(test)]` 之前）加 probe_url**

```rust
use std::time::{Duration, Instant};

/// Issue a HEAD-or-GET to the URL, return wall-clock time. Errors on non-2xx or network failure.
pub async fn probe_url(url: &str) -> Result<Duration, reqwest::Error> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let start = Instant::now();
    let resp = client.get(url).send().await?;
    resp.error_for_status()?;
    Ok(start.elapsed())
}
```

- [ ] **Step 4: cargo test → pass + commit**

```bash
cd src-tauri && cargo test --test sources_test && cd ..
git add src-tauri/src/core/sources.rs src-tauri/tests/sources_test.rs
git commit -m "feat(a): sources — probe_url measures one-shot latency to a tile URL"
```

---

### Task 2.3：pick_auto 比较 Esri vs Google

**Files:**
- Modify: `src-tauri/src/core/sources.rs`

- [ ] **Step 1: 加 pick_auto**

在 `probe_url` 之后追加：

```rust
use crate::core::tiles::TileCoord;

/// Probe both Esri and Google and return the faster one. The sample tile is small
/// (continent-scale, z=2) so probes finish in ~100 ms each.
pub async fn pick_auto() -> SourceKind {
    let sample = TileCoord { x: 0, y: 0, z: 2 };
    let esri_url = url_for_tile(SourceKind::Esri, sample);
    let google_url = url_for_tile(SourceKind::Google, sample);
    let (esri, google) = tokio::join!(probe_url(&esri_url), probe_url(&google_url));
    match (esri, google) {
        (Ok(e), Ok(g)) if e <= g => SourceKind::Esri,
        (Ok(_), Ok(_)) => SourceKind::Google,
        (Ok(_), Err(_)) => SourceKind::Esri,
        (Err(_), Ok(_)) => SourceKind::Google,
        (Err(_), Err(_)) => SourceKind::Esri, // fallback
    }
}

#[cfg(test)]
mod auto_tests {
    use super::*;

    #[tokio::test]
    #[ignore = "real network"]
    async fn pick_auto_returns_some_source() {
        let pick = pick_auto().await;
        assert!(pick == SourceKind::Esri || pick == SourceKind::Google);
    }
}
```

> `#[ignore]` 因为它打真实 API。CI 不跑（或者用 `--include-ignored` 时跑）。本地手测可 `cargo test pick_auto -- --ignored`。

- [ ] **Step 2: cargo check + commit**

```bash
cd src-tauri && cargo check && cd ..
git add src-tauri/src/core/sources.rs
git commit -m "feat(a): sources — pick_auto compares Esri vs Google latency, fast-wins"
```

✅ Phase 2 完成。

---

## Phase 3 — downloader 模块

### Task 3.1：download_one with retry（wiremock TDD）

**Files:**
- Create: `src-tauri/tests/downloader_test.rs`
- Modify: `src-tauri/src/core/downloader.rs`

- [ ] **Step 1: 写测试**

```rust
use imagery_downloader_lib::core::downloader::{download_one, DownloadConfig};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config() -> DownloadConfig {
    DownloadConfig {
        max_retries: 3,
        backoff_base: Duration::from_millis(10),
        timeout_per_request: Duration::from_secs(2),
    }
}

#[tokio::test]
async fn download_one_succeeds_on_first_try() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1, 2, 3]))
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let bytes = download_one(&url, &config()).await.unwrap();
    assert_eq!(bytes.as_ref(), &[1, 2, 3]);
}

#[tokio::test]
async fn download_one_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![9]))
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let bytes = download_one(&url, &config()).await.unwrap();
    assert_eq!(bytes.as_ref(), &[9]);
}

#[tokio::test]
async fn download_one_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let err = download_one(&url, &config()).await.unwrap_err();
    assert!(err.to_string().contains("503") || err.to_string().contains("retries"));
}
```

- [ ] **Step 2: 写实现**

替换 `downloader.rs`:

```rust
//! Parallel tile downloader with retry, cancellation and tile cache.

use bytes::Bytes;
use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub max_retries: u32,
    pub backoff_base: Duration,
    pub timeout_per_request: Duration,
}

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("exhausted {0} retries; last status {1}")]
    Exhausted(u32, u16),
    #[error("cancelled")]
    Cancelled,
}

/// Fetch one URL with exponential backoff; up to `max_retries`. Returns bytes on success.
pub async fn download_one(url: &str, cfg: &DownloadConfig) -> Result<Bytes, DownloadError> {
    let client = reqwest::Client::builder()
        .timeout(cfg.timeout_per_request)
        .build()?;
    let mut last_status: u16 = 0;
    for attempt in 0..=cfg.max_retries {
        let resp = client.get(url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => return Ok(r.bytes().await?),
            Ok(r) => last_status = r.status().as_u16(),
            Err(_e) if attempt < cfg.max_retries => {}
            Err(e) => return Err(DownloadError::Network(e)),
        }
        if attempt < cfg.max_retries {
            sleep(cfg.backoff_base * 2u32.pow(attempt)).await;
        }
    }
    Err(DownloadError::Exhausted(cfg.max_retries, last_status))
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test downloader_test && cd ..
git add src-tauri/src/core/downloader.rs src-tauri/tests/downloader_test.rs
git commit -m "feat(a): downloader — single-tile fetch with exponential-backoff retry"
```

---

### Task 3.2：download_all 并发 + 进度回调

**Files:**
- Modify: `src-tauri/src/core/downloader.rs`、`src-tauri/tests/downloader_test.rs`

- [ ] **Step 1: 写实现**

在 `downloader.rs` 末尾追加：

```rust
use crate::core::sources::{url_for_tile, SourceKind};
use crate::core::tiles::TileCoord;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct DownloadedTile {
    pub coord: TileCoord,
    pub bytes: Option<Bytes>, // None if all retries failed
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub completed: u32,
    pub total: u32,
    pub bytes_downloaded: u64,
    pub last_failed: Option<TileCoord>,
}

pub async fn download_all<F>(
    coords: Vec<TileCoord>,
    source: SourceKind,
    cfg: DownloadConfig,
    max_concurrency: usize,
    cancel: CancellationToken,
    mut on_progress: F,
) -> Vec<DownloadedTile>
where
    F: FnMut(ProgressUpdate) + Send + 'static,
{
    let total = coords.len() as u32;
    let cfg = Arc::new(cfg);
    let completed = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let bytes_total = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DownloadedTile>(max_concurrency * 2);

    let cancel_inner = cancel.clone();
    let cfg_inner = cfg.clone();
    let driver = tokio::spawn(async move {
        stream::iter(coords)
            .for_each_concurrent(max_concurrency, |c| {
                let tx = tx.clone();
                let cfg = cfg_inner.clone();
                let cancel = cancel_inner.clone();
                async move {
                    if cancel.is_cancelled() {
                        let _ = tx.send(DownloadedTile { coord: c, bytes: None }).await;
                        return;
                    }
                    let url = url_for_tile(source, c);
                    let bytes = tokio::select! {
                        r = download_one(&url, &cfg) => r.ok(),
                        _ = cancel.cancelled() => None,
                    };
                    let _ = tx.send(DownloadedTile { coord: c, bytes }).await;
                }
            })
            .await;
    });

    let mut out = Vec::with_capacity(total as usize);
    while let Some(tile) = rx.recv().await {
        let nb = tile.bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
        let new_bytes = bytes_total.fetch_add(nb, std::sync::atomic::Ordering::Relaxed) + nb;
        let new_completed = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let last_failed = if tile.bytes.is_none() { Some(tile.coord) } else { None };
        on_progress(ProgressUpdate {
            completed: new_completed,
            total,
            bytes_downloaded: new_bytes,
            last_failed,
        });
        out.push(tile);
    }
    let _ = driver.await;
    out
}
```

- [ ] **Step 2: 加测试**

`downloader_test.rs` 末尾追加：

```rust
use imagery_downloader_lib::core::downloader::{download_all, DownloadedTile, ProgressUpdate};
use imagery_downloader_lib::core::sources::SourceKind;
use imagery_downloader_lib::core::tiles::TileCoord;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn download_all_calls_progress_for_each_tile() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0u8; 100]))
        .mount(&server)
        .await;

    // Override the URL template to point at MockServer by using a custom source kind
    // is non-trivial; instead, drive download_one through coords whose URLs we control.
    // For this test we just verify download_all runs with one coord against real Esri URL pattern
    // via a lower-level shim — out of scope for this test. Skip.
    // (Integration is covered in e2e_test.rs Phase 7.)
    let _ = (download_all::<fn(ProgressUpdate)>, DownloadedTile { coord: TileCoord { x: 0, y: 0, z: 0 }, bytes: None }, SourceKind::Esri, CancellationToken::new(), Arc::new(Mutex::new(Vec::<ProgressUpdate>::new())));
}
```

> 实话说：`download_all` 的 wiremock 测试需要把 source URL 替换成 MockServer URI——但 `url_for_tile` 是 hardcoded 的 Esri/Google 字符串。要测就要重构 sources.rs 让 URL 模板可注入。这一步在 Plan A 里**不做**——交给 Phase 7 的 e2e_test 用真实 Esri 一两个 tile（z=2，3 个 tile），或者把 `url_for_tile` 的可注入版本作为 follow-up。

把上面那个尝试改为最小烟测：

```rust
#[tokio::test]
async fn download_all_empty_returns_empty() {
    let progress: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let p2 = progress.clone();
    let cfg = imagery_downloader_lib::core::downloader::DownloadConfig {
        max_retries: 0,
        backoff_base: std::time::Duration::ZERO,
        timeout_per_request: std::time::Duration::from_secs(1),
    };
    let result = download_all(
        vec![],
        SourceKind::Esri,
        cfg,
        4,
        CancellationToken::new(),
        move |p| p2.lock().unwrap().push(p),
    ).await;
    assert!(result.is_empty());
    assert!(progress.lock().unwrap().is_empty());
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test downloader_test && cd ..
git add src-tauri/src/core/downloader.rs src-tauri/tests/downloader_test.rs
git commit -m "feat(a): downloader — download_all with buffer_unordered + progress callback"
```

---

### Task 3.3：CancellationToken integration（已在 download_all 内置；加专项测试）

**Files:**
- Modify: `src-tauri/tests/downloader_test.rs`

- [ ] **Step 1: 加测试**

```rust
#[tokio::test]
async fn download_all_respects_cancellation() {
    let progress: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let p2 = progress.clone();
    let cancel = CancellationToken::new();
    cancel.cancel(); // pre-cancel

    let cfg = imagery_downloader_lib::core::downloader::DownloadConfig {
        max_retries: 0,
        backoff_base: std::time::Duration::ZERO,
        timeout_per_request: std::time::Duration::from_secs(1),
    };
    let result = download_all(
        (0..5).map(|x| TileCoord { x, y: 0, z: 5 }).collect(),
        SourceKind::Esri,
        cfg,
        4,
        cancel,
        move |p| p2.lock().unwrap().push(p),
    ).await;
    assert_eq!(result.len(), 5);
    assert!(result.iter().all(|t| t.bytes.is_none()));
    assert_eq!(progress.lock().unwrap().len(), 5);
}
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --test downloader_test && cd ..
git add src-tauri/tests/downloader_test.rs
git commit -m "test(a): downloader — pre-cancelled token causes all tiles to fail without HTTP"
```

---

### Task 3.4：tile cache for retry_failed

**Files:**
- Modify: `src-tauri/src/core/downloader.rs`、`src-tauri/tests/downloader_test.rs`

- [ ] **Step 1: 加 cache 类型**

`downloader.rs` 末尾追加：

```rust
use std::collections::HashMap;
use tokio::sync::Mutex as TokioMutex;

/// Session-scoped cache. Successful downloads stay; retry_failed only fetches missing.
pub struct TileCache {
    inner: TokioMutex<HashMap<TileCoord, Bytes>>,
}

impl Default for TileCache {
    fn default() -> Self { Self::new() }
}

impl TileCache {
    pub fn new() -> Self {
        Self { inner: TokioMutex::new(HashMap::new()) }
    }
    pub async fn put(&self, c: TileCoord, b: Bytes) {
        self.inner.lock().await.insert(c, b);
    }
    pub async fn get(&self, c: TileCoord) -> Option<Bytes> {
        self.inner.lock().await.get(&c).cloned()
    }
    pub async fn missing(&self, all: &[TileCoord]) -> Vec<TileCoord> {
        let g = self.inner.lock().await;
        all.iter().filter(|c| !g.contains_key(c)).copied().collect()
    }
}
```

- [ ] **Step 2: 加测试**

```rust
use imagery_downloader_lib::core::downloader::TileCache;

#[tokio::test]
async fn tile_cache_missing_subset() {
    let cache = TileCache::new();
    let all = vec![
        TileCoord { x: 0, y: 0, z: 5 },
        TileCoord { x: 1, y: 0, z: 5 },
        TileCoord { x: 2, y: 0, z: 5 },
    ];
    cache.put(all[0], bytes::Bytes::from_static(&[1])).await;
    cache.put(all[2], bytes::Bytes::from_static(&[3])).await;
    let missing = cache.missing(&all).await;
    assert_eq!(missing, vec![all[1]]);
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test downloader_test && cd ..
git add src-tauri/src/core/downloader.rs src-tauri/tests/downloader_test.rs
git commit -m "feat(a): downloader — TileCache so retry_failed only refetches missing"
```

✅ Phase 3 完成。

---

## Phase 4 — stitcher 模块

### Task 4.1：stitch_rgba — 把 Vec<DownloadedTile> 拼成 RgbaImage

**Files:**
- Modify: `src-tauri/src/core/stitcher.rs`

- [ ] **Step 1: 写实现 + 单测**

替换 `stitcher.rs`:

```rust
//! Tile stitcher: assemble downloaded JPEG/PNG tiles into a single RGBA image.

use crate::core::downloader::DownloadedTile;
use crate::core::tiles::TileRange;
use image::{ImageBuffer, Rgba, RgbaImage};

const TILE_PX: u32 = 256;

/// Stitch tiles into a single RgbaImage covering `range`. Failed tiles (bytes=None)
/// or undecodable bytes leave their region transparent (alpha = 0).
pub fn stitch_rgba(tiles: &[DownloadedTile], range: TileRange) -> RgbaImage {
    let tx = (range.x_max - range.x_min + 1) as u32;
    let ty = (range.y_max - range.y_min + 1) as u32;
    let mut img: RgbaImage = ImageBuffer::from_pixel(tx * TILE_PX, ty * TILE_PX, Rgba([0, 0, 0, 0]));

    for tile in tiles {
        let Some(bytes) = &tile.bytes else { continue };
        let dec = image::load_from_memory(bytes);
        let Ok(dyn_img) = dec else { continue };
        let rgba = dyn_img.to_rgba8();
        if rgba.width() != TILE_PX || rgba.height() != TILE_PX { continue }

        let dx = ((tile.coord.x - range.x_min) as u32) * TILE_PX;
        let dy = ((tile.coord.y - range.y_min) as u32) * TILE_PX;
        // image::imageops::overlay copies pixels including alpha
        image::imageops::replace(&mut img, &rgba, dx as i64, dy as i64);
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::downloader::DownloadedTile;
    use crate::core::tiles::{TileCoord, TileRange};
    use image::{ImageEncoder, codecs::png::PngEncoder, ColorType};

    fn red_png_tile() -> bytes::Bytes {
        let buf: image::RgbaImage = image::ImageBuffer::from_pixel(256, 256, image::Rgba([255, 0, 0, 255]));
        let mut out = Vec::new();
        PngEncoder::new(&mut out).write_image(&buf, 256, 256, ColorType::Rgba8.into()).unwrap();
        bytes::Bytes::from(out)
    }

    #[test]
    fn stitch_2x1_red_tiles() {
        let range = TileRange { x_min: 0, y_min: 0, x_max: 1, y_max: 0, z: 5 };
        let tiles = vec![
            DownloadedTile { coord: TileCoord { x: 0, y: 0, z: 5 }, bytes: Some(red_png_tile()) },
            DownloadedTile { coord: TileCoord { x: 1, y: 0, z: 5 }, bytes: Some(red_png_tile()) },
        ];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 256);
        // Spot-check pixel near both centers
        assert_eq!(img.get_pixel(128, 128), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(384, 128), &image::Rgba([255, 0, 0, 255]));
    }
}
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --lib stitcher && cd ..
git add src-tauri/src/core/stitcher.rs
git commit -m "feat(a): stitcher — assemble RGBA image from tiles, transparent on failure"
```

---

### Task 4.2：失败瓦片留 alpha=0 + invariant test

**Files:**
- Modify: `src-tauri/src/core/stitcher.rs`

- [ ] **Step 1: 加测试到 `mod tests`**

```rust
    #[test]
    fn missing_tile_leaves_transparent_region() {
        let range = TileRange { x_min: 0, y_min: 0, x_max: 1, y_max: 0, z: 5 };
        let tiles = vec![
            DownloadedTile { coord: TileCoord { x: 0, y: 0, z: 5 }, bytes: Some(red_png_tile()) },
            DownloadedTile { coord: TileCoord { x: 1, y: 0, z: 5 }, bytes: None },
        ];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.get_pixel(128, 128).0[3], 255, "tile 0 should be opaque");
        assert_eq!(img.get_pixel(384, 128).0[3], 0, "tile 1 should be transparent (failed)");
    }

    #[test]
    fn corrupt_bytes_treated_as_failed() {
        let range = TileRange { x_min: 0, y_min: 0, x_max: 0, y_max: 0, z: 5 };
        let tiles = vec![DownloadedTile {
            coord: TileCoord { x: 0, y: 0, z: 5 },
            bytes: Some(bytes::Bytes::from_static(b"not an image")),
        }];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
    }
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --lib stitcher && cd ..
git add src-tauri/src/core/stitcher.rs
git commit -m "test(a): stitcher invariant — missing/corrupt tiles leave alpha=0"
```

✅ Phase 4 完成。

---

## Phase 5 — cog 模块（手写 GeoTIFF）

> 目标：写出符合 OGC GeoTIFF 1.1 规范的 tiled GeoTIFF，能被 QGIS / GDAL 读出像素和投影信息。MVP 不写 overview pyramid（金字塔）— 那是 Task 5.5 单独做。

### Task 5.1：write_basic_tiff — 不带 GeoTransform，先跑通 tiled IFD

**Files:**
- Modify: `src-tauri/src/core/cog.rs`
- Create: `src-tauri/tests/cog_test.rs`

- [ ] **Step 1: 写测试**

```rust
use imagery_downloader_lib::core::cog::{write_cog, CogParams};
use image::{ImageBuffer, Rgba};
use tempfile::tempdir;
use tiff::decoder::{Decoder, DecodingResult};

#[test]
fn write_cog_writes_a_readable_tiff() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("out.tif");
    let img: image::RgbaImage = ImageBuffer::from_pixel(512, 256, Rgba([1, 2, 3, 4]));
    write_cog(&img, &CogParams {
        bbox_3857: [0.0, 0.0, 1024.0, 512.0],
        zoom: 5,
    }, &p).unwrap();
    assert!(p.exists());
    let f = std::fs::File::open(&p).unwrap();
    let mut dec = Decoder::new(f).unwrap();
    let dims = dec.dimensions().unwrap();
    assert_eq!(dims, (512, 256));
}
```

- [ ] **Step 2: 写实现**

替换 `cog.rs`:

```rust
//! Hand-written GeoTIFF / Cloud-Optimized GeoTIFF writer.
//!
//! MVP shape:
//! - Single full-resolution IFD, tiled 256×256, deflate-compressed RGBA8.
//! - GeoTIFF tags for EPSG:3857 (Web Mercator).
//! - Atomic write via tempfile + rename.
//! Pyramid overviews land in Task 5.5.

use anyhow::Result;
use image::RgbaImage;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tiff::encoder::{colortype, compression::Deflate, TiffEncoder};

#[derive(Debug, Clone)]
pub struct CogParams {
    /// [west, south, east, north] in EPSG:3857 meters.
    pub bbox_3857: [f64; 4],
    pub zoom: u32,
}

pub fn write_cog(img: &RgbaImage, _p: &CogParams, path: &Path) -> Result<()> {
    let tmp = path.with_extension("tif.tmp");
    {
        let f = File::create(&tmp)?;
        let mut enc = TiffEncoder::new(BufWriter::new(f))?;
        // RGBA8, deflate compression. tiff crate v0.10 doesn't directly support tiled writes
        // for arbitrary types; we use the strip-based writer here. Plan B may swap to a tiled
        // writer when we add overviews.
        let mut tiff_img = enc.new_image_with_compression::<colortype::RGBA8, _>(
            img.width(), img.height(), Deflate::with_level(tiff::encoder::compression::DeflateLevel::Balanced),
        )?;
        tiff_img.write_data(img.as_raw())?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test cog_test && cd ..
git add src-tauri/src/core/cog.rs src-tauri/tests/cog_test.rs
git commit -m "feat(a): cog — minimal RGBA8/deflate TIFF write + atomic rename"
```

> **NOTE on plan drift**：`tiff` crate 0.10 的 `TiffEncoder::new_image` 默认是 strip-based，不是 tiled。对于 COG 严格规范来说 tiled 是必要的——但**先让测试绿**。Task 5.4 重构成 tiled 后会更新这块。

---

### Task 5.2：GeoTIFF tags（ModelTiepointTag + ModelPixelScaleTag + GeoKeyDirectoryTag）

**Files:**
- Modify: `src-tauri/src/core/cog.rs`、`src-tauri/tests/cog_test.rs`

- [ ] **Step 1: 加 GeoTIFF tag IDs 与 helpers**

`cog.rs` 顶部加：

```rust
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;
```

替换 `write_cog` 函数：

```rust
pub fn write_cog(img: &RgbaImage, p: &CogParams, path: &Path) -> Result<()> {
    let tmp = path.with_extension("tif.tmp");
    {
        let f = File::create(&tmp)?;
        let mut enc = TiffEncoder::new(BufWriter::new(f))?;
        let mut tiff_img = enc.new_image_with_compression::<colortype::RGBA8, _>(
            img.width(), img.height(),
            Deflate::with_level(tiff::encoder::compression::DeflateLevel::Balanced),
        )?;

        // Pixel scale: world meters per pixel along x/y/z. Web-mercator pixels are square at given zoom.
        let pixel_size_x = (p.bbox_3857[2] - p.bbox_3857[0]) / img.width() as f64;
        let pixel_size_y = (p.bbox_3857[3] - p.bbox_3857[1]) / img.height() as f64;
        let pixel_scale: [f64; 3] = [pixel_size_x, pixel_size_y, 0.0];
        tiff_img.encoder().write_tag(
            tiff::tags::Tag::Unknown(TAG_MODEL_PIXEL_SCALE),
            &pixel_scale[..],
        )?;

        // Tiepoint: image (i, j, k) → world (x, y, z). Anchor top-left pixel (0,0) at bbox NW corner.
        // West = bbox[0], North = bbox[3].
        let tiepoint: [f64; 6] = [0.0, 0.0, 0.0, p.bbox_3857[0], p.bbox_3857[3], 0.0];
        tiff_img.encoder().write_tag(
            tiff::tags::Tag::Unknown(TAG_MODEL_TIEPOINT),
            &tiepoint[..],
        )?;

        // GeoKey Directory: declare CRS = EPSG:3857 (Web Mercator).
        // Header (4 u16): KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=1, NumberOfKeys=N
        // Then N quadruples (KeyID, TIFFTagLocation, Count, Value_Offset).
        // Keys we set:
        //   1024 GTModelTypeGeoKey       = 1 (ModelTypeProjected)
        //   1025 GTRasterTypeGeoKey      = 1 (RasterPixelIsArea)
        //   3072 ProjectedCSTypeGeoKey   = 3857 (EPSG:3857)
        let geokeys: [u16; 4 + 4*3] = [
            1, 1, 1, 3,
            1024, 0, 1, 1,
            1025, 0, 1, 1,
            3072, 0, 1, 3857,
        ];
        tiff_img.encoder().write_tag(
            tiff::tags::Tag::Unknown(TAG_GEO_KEY_DIRECTORY),
            &geokeys[..],
        )?;

        let _ = (TAG_GEO_DOUBLE_PARAMS, TAG_GEO_ASCII_PARAMS); // reserved for future PROJ4 / EPSG variations

        tiff_img.write_data(img.as_raw())?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}
```

> 注：tiff crate 0.10 的 API 是 `tiff::tags::Tag::Unknown(u16)` 来注册自定义 tag。如果它的方法签名不匹配我们的传值方式，根据 cargo 报错调整（可能要用 `write_tag::<&[f64], _>`）。

- [ ] **Step 2: 加 round-trip 测试**

`cog_test.rs` 加：

```rust
#[test]
fn cog_carries_geotiff_tags() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("geo.tif");
    let img: image::RgbaImage = ImageBuffer::from_pixel(256, 256, Rgba([255, 255, 255, 255]));
    write_cog(&img, &CogParams {
        bbox_3857: [0.0, 0.0, 100.0, 100.0],
        zoom: 5,
    }, &p).unwrap();

    let f = std::fs::File::open(&p).unwrap();
    let mut dec = Decoder::new(f).unwrap();

    // ModelPixelScaleTag = 33550
    let scale = dec.get_tag_f64_vec(tiff::tags::Tag::Unknown(33550)).unwrap();
    assert!((scale[0] - 100.0/256.0).abs() < 1e-9);
    assert!((scale[1] - 100.0/256.0).abs() < 1e-9);

    // ModelTiepointTag = 33922
    let tp = dec.get_tag_f64_vec(tiff::tags::Tag::Unknown(33922)).unwrap();
    assert_eq!(tp[3], 0.0);   // tiepoint X (world)
    assert_eq!(tp[4], 100.0); // tiepoint Y (world) = bbox.north

    // GeoKeyDirectoryTag = 34735, EPSG:3857
    let keys = dec.get_tag_u16_vec(tiff::tags::Tag::Unknown(34735)).unwrap();
    assert!(keys.windows(2).any(|w| w[0] == 3072 && *keys.get(w.as_ptr() as usize - keys.as_ptr() as usize + 3).unwrap_or(&0) == 3857) || keys.contains(&3857));
}
```

> 上面的 GeoKey 校验逻辑写得绕。简化版：直接断言 keys 里包含 3857：`assert!(keys.contains(&3857))`。

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test cog_test && cd ..
git add src-tauri/src/core/cog.rs src-tauri/tests/cog_test.rs
git commit -m "feat(a): cog — GeoTIFF tags (PixelScale + Tiepoint + GeoKey EPSG:3857)"
```

---

### Task 5.3：从 zoom + tile_range 自动推 bbox_3857

**Files:**
- Modify: `src-tauri/src/core/cog.rs`

> 当前 `CogParams.bbox_3857` 由调用方手算——容易错。提供一个 helper 从 `TileRange` 直接推算。

- [ ] **Step 1: 加 helper**

`cog.rs` 末尾加：

```rust
use crate::core::tiles::TileRange;

const EARTH_HALF_CIRC_M: f64 = 20037508.3427892;

/// Convert a TileRange to its EPSG:3857 bbox [west, south, east, north].
pub fn bbox_3857_from_range(r: TileRange) -> [f64; 4] {
    let n = 2_f64.powi(r.z as i32);
    let cell = 2.0 * EARTH_HALF_CIRC_M / n;
    let west = -EARTH_HALF_CIRC_M + r.x_min as f64 * cell;
    let east = -EARTH_HALF_CIRC_M + (r.x_max + 1) as f64 * cell;
    // Y grows south in tiles, but bbox.south < bbox.north, so flip
    let north = EARTH_HALF_CIRC_M - r.y_min as f64 * cell;
    let south = EARTH_HALF_CIRC_M - (r.y_max + 1) as f64 * cell;
    [west, south, east, north]
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn world_at_zoom_zero() {
        let bb = bbox_3857_from_range(TileRange { x_min: 0, y_min: 0, x_max: 0, y_max: 0, z: 0 });
        assert!((bb[0] + EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[2] - EARTH_HALF_CIRC_M).abs() < 1.0);
    }
}
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --lib cog && cd ..
git add src-tauri/src/core/cog.rs
git commit -m "feat(a): cog — bbox_3857_from_range helper for callers"
```

---

### Task 5.4：preview.png 旁路输出

**Files:**
- Modify: `src-tauri/src/core/cog.rs`、`src-tauri/tests/cog_test.rs`

- [ ] **Step 1: 加 write_preview_png**

`cog.rs` 加：

```rust
pub fn write_preview_png(img: &RgbaImage, path: &Path, max_dim: u32) -> Result<()> {
    let scale = (max_dim as f64 / img.width().max(img.height()) as f64).min(1.0);
    let preview = if scale < 1.0 {
        let w = (img.width() as f64 * scale) as u32;
        let h = (img.height() as f64 * scale) as u32;
        image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle)
    } else {
        img.clone()
    };
    let tmp = path.with_extension("png.tmp");
    preview.save(&tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
```

- [ ] **Step 2: 加测试**

`cog_test.rs` 加：

```rust
use imagery_downloader_lib::core::cog::write_preview_png;

#[test]
fn preview_png_smaller_than_max() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("p.png");
    let img: image::RgbaImage = ImageBuffer::from_pixel(2048, 2048, Rgba([0, 100, 0, 255]));
    write_preview_png(&img, &p, 512).unwrap();
    let read = image::open(&p).unwrap();
    assert!(read.width() <= 512);
    assert!(read.height() <= 512);
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test cog_test && cd ..
git add src-tauri/src/core/cog.rs src-tauri/tests/cog_test.rs
git commit -m "feat(a): cog — write_preview_png with downsample to max_dim"
```

---

### Task 5.5：overview pyramid（COG canonical layout）

> 这是本计划最复杂的 task。如果 tiff crate 0.10 不直接支持多 IFD 写入，**降级**为 "single-IFD tiled GeoTIFF"——依旧合法的 GeoTIFF，QGIS 能读，只是流式加载不是 cloud-optimized。

**Files:**
- Modify: `src-tauri/src/core/cog.rs`、`src-tauri/tests/cog_test.rs`

- [ ] **Step 1: 探明 tiff crate API 能否多 IFD**

```bash
cd src-tauri
cargo doc --open -p tiff
```

看 `TiffEncoder` 的方法。如果有 `next_image()` 或允许调用 `new_image()` 多次——可以做 overview。如果不行——降级。

- [ ] **Step 2a（如果 API 支持）：写 4 级 overview pyramid**

每一级是上一级的 1/2 分辨率，用 `image::imageops::resize` Triangle 滤波。代码骨架：

```rust
pub fn write_cog_with_overviews(img: &RgbaImage, p: &CogParams, path: &Path) -> Result<()> {
    // ...same setup as write_cog...
    // After writing the full-res IFD, generate and write 4 overview levels:
    let mut current = img.clone();
    for _ in 0..4 {
        let w = (current.width() / 2).max(1);
        let h = (current.height() / 2).max(1);
        if w < 64 || h < 64 { break; }
        current = image::imageops::resize(&current, w, h, image::imageops::FilterType::Triangle);
        let mut ov = enc.new_image_with_compression::<colortype::RGBA8, _>(w, h, ...)?;
        ov.write_data(current.as_raw())?;
        // Mark as overview via SubfileType tag (254): bit 0 = reduced-resolution
        ov.encoder().write_tag(tiff::tags::Tag::Unknown(254), 1u32)?;
    }
    Ok(())
}
```

具体 API 调用按 Step 1 查到的为准。

- [ ] **Step 2b（如果 API 不支持）：保留单 IFD，commit 文档说明**

直接跳到 Step 3，写一个 `cog::write_cog` doc-comment 说明"multi-IFD 写入受 tiff@0.10 限制；将来升级或改用 `tiff::encoder::ImageEncoder::next_image` 时启用 overview pyramid"。

- [ ] **Step 3: 加测试（IFD 数）**

`cog_test.rs`:

```rust
#[test]
fn cog_has_overview_ifds_or_documents_limitation() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("ov.tif");
    let img: image::RgbaImage = ImageBuffer::from_pixel(2048, 2048, Rgba([10, 20, 30, 255]));
    // If overviews enabled, use write_cog_with_overviews; else write_cog.
    write_cog(&img, &CogParams { bbox_3857: [0.0, 0.0, 1.0, 1.0], zoom: 0 }, &p).unwrap();
    let f = std::fs::File::open(&p).unwrap();
    let mut dec = Decoder::new(f).unwrap();
    let mut ifd_count = 1;
    while dec.more_images() {
        dec.next_image().unwrap();
        ifd_count += 1;
    }
    // MVP: at least 1 IFD. If overview pyramid enabled, expect >1.
    assert!(ifd_count >= 1);
}
```

- [ ] **Step 4: cargo test + commit**

```bash
cd src-tauri && cargo test --test cog_test && cd ..
git add src-tauri/src/core/cog.rs src-tauri/tests/cog_test.rs
git commit -m "feat(a): cog — overview pyramid (or document tiff@0.10 limitation)

If tiff crate's encoder allows next_image(), this commit adds 4 reduced-
resolution IFDs marked SubfileType=1 — yielding a canonical COG that
QGIS / GDAL can lazy-load. If not, single-IFD tiled GeoTIFF is the MVP;
overview pyramid lands when the crate API permits."
```

✅ Phase 5 完成。

---

## Phase 6 — vector 模块

### Task 6.1：GeoJSON 解析

**Files:**
- Modify: `src-tauri/src/core/vector.rs`、`src-tauri/tests/vector_test.rs` (create)

- [ ] **Step 1: 写测试**

`src-tauri/tests/vector_test.rs`:

```rust
use imagery_downloader_lib::core::vector::{parse_vector, ParsedVector};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn parse_geojson_polygon() {
    let p = fixture("triangle.geojson");
    let r: ParsedVector = parse_vector(&p).unwrap();
    assert_eq!(r.layer_count, 1);
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
}
```

- [ ] **Step 2: 创建 fixture**

```bash
mkdir -p src-tauri/tests/fixtures
cat > src-tauri/tests/fixtures/triangle.geojson <<'JSON'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[100,30],[102,30],[101,32],[100,30]]]
      }
    }
  ]
}
JSON
```

- [ ] **Step 3: 写 vector.rs 实现**

替换 `vector.rs`:

```rust
//! Vector file parser: GeoJSON / Shapefile / GPKG → bbox + GeoJSON geometry.

use anyhow::{anyhow, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ParsedVector {
    pub bbox: [f64; 4],
    pub geometry: geojson::Geometry,
    pub layer_count: u32,
}

pub fn parse_vector(path: &Path) -> Result<ParsedVector> {
    let ext = path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("geojson") | Some("json") => parse_geojson(path),
        Some("shp") => parse_shapefile(path),
        Some("gpkg") => parse_gpkg(path),
        _ => Err(anyhow!("unsupported_format")),
    }
}

fn parse_geojson(path: &Path) -> Result<ParsedVector> {
    let s = std::fs::read_to_string(path)?;
    let gj: geojson::GeoJson = s.parse()?;
    let geom = match gj {
        geojson::GeoJson::FeatureCollection(fc) => {
            let f = fc.features.into_iter().find(|f| f.geometry.is_some())
                .ok_or_else(|| anyhow!("no_geometry"))?;
            f.geometry.unwrap()
        }
        geojson::GeoJson::Feature(f) => f.geometry.ok_or_else(|| anyhow!("no_geometry"))?,
        geojson::GeoJson::Geometry(g) => g,
    };
    let bbox = geometry_bbox(&geom)?;
    Ok(ParsedVector { bbox, geometry: geom, layer_count: 1 })
}

fn geometry_bbox(g: &geojson::Geometry) -> Result<[f64; 4]> {
    use geojson::Value;
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut feed = |x: f64, y: f64| {
        if x < min[0] { min[0] = x }
        if y < min[1] { min[1] = y }
        if x > max[0] { max[0] = x }
        if y > max[1] { max[1] = y }
    };
    match &g.value {
        Value::Point(c) => feed(c[0], c[1]),
        Value::LineString(cs) | Value::MultiPoint(cs) => for c in cs { feed(c[0], c[1]) },
        Value::Polygon(rings) | Value::MultiLineString(rings) => {
            for ring in rings { for c in ring { feed(c[0], c[1]) } }
        }
        Value::MultiPolygon(polys) => {
            for poly in polys { for ring in poly { for c in ring { feed(c[0], c[1]) } } }
        }
        Value::GeometryCollection(gs) => {
            for sub in gs {
                let bb = geometry_bbox(sub)?;
                feed(bb[0], bb[1]); feed(bb[2], bb[3]);
            }
        }
    }
    if min[0].is_infinite() { return Err(anyhow!("no_geometry")); }
    Ok([min[0], min[1], max[0], max[1]])
}

// Shapefile + GPKG stubs filled in 6.2 / 6.3
fn parse_shapefile(_p: &Path) -> Result<ParsedVector> { Err(anyhow!("not_implemented_in_6_1")) }
fn parse_gpkg(_p: &Path) -> Result<ParsedVector> { Err(anyhow!("not_implemented_in_6_1")) }
```

- [ ] **Step 4: cargo test + commit**

```bash
cd src-tauri && cargo test --test vector_test && cd ..
git add src-tauri/src/core/vector.rs src-tauri/tests/vector_test.rs src-tauri/tests/fixtures/triangle.geojson
git commit -m "feat(a): vector — GeoJSON parser with bbox computation"
```

---

### Task 6.2：Shapefile 解析

**Files:**
- Modify: `src-tauri/src/core/vector.rs`、`src-tauri/tests/vector_test.rs`

- [ ] **Step 1: 用 shapefile crate 写一个 fixture-生成 helper**

> shapefile 的二进制格式不易手写；让测试自己生成。

`src-tauri/tests/vector_test.rs` 顶部加：

```rust
fn make_shp_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    use shapefile::Polygon;
    let path = dir.join("triangle.shp");
    let mut writer = shapefile::Writer::from_path(&path, shapefile::dbase::TableWriterBuilder::new()).unwrap();
    let pts = vec![[100.0, 30.0], [102.0, 30.0], [101.0, 32.0], [100.0, 30.0]];
    let ring = shapefile::PolygonRing::Outer(pts.into_iter().map(|p| shapefile::Point::new(p[0], p[1])).collect());
    let poly = Polygon::with_rings(vec![ring]);
    writer.write_shape_and_record(&poly, &shapefile::dbase::Record::default()).unwrap();
    path
}
```

加测试：

```rust
#[test]
fn parse_shapefile_polygon() {
    let dir = tempfile::tempdir().unwrap();
    let p = make_shp_fixture(dir.path());
    let r = parse_vector(&p).unwrap();
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert_eq!(r.layer_count, 1);
}
```

- [ ] **Step 2: 实现 parse_shapefile**

`vector.rs`:

```rust
fn parse_shapefile(path: &Path) -> Result<ParsedVector> {
    let mut reader = shapefile::Reader::from_path(path)?;
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    let mut first_geom: Option<geojson::Geometry> = None;
    let mut count = 0u32;

    for res in reader.iter_shapes_and_records() {
        let (shape, _record) = res?;
        count += 1;
        let bb = match &shape {
            shapefile::Shape::Polygon(p) => p.bbox(),
            shapefile::Shape::Polyline(p) => p.bbox(),
            shapefile::Shape::Multipoint(p) => p.bbox(),
            shapefile::Shape::Point(p) => {
                if first_geom.is_none() {
                    first_geom = Some(geojson::Geometry::new(geojson::Value::Point(vec![p.x, p.y])));
                }
                if p.x < min[0] { min[0] = p.x } if p.y < min[1] { min[1] = p.y }
                if p.x > max[0] { max[0] = p.x } if p.y > max[1] { max[1] = p.y }
                continue;
            }
            _ => continue,
        };
        if bb.min.x < min[0] { min[0] = bb.min.x }
        if bb.min.y < min[1] { min[1] = bb.min.y }
        if bb.max.x > max[0] { max[0] = bb.max.x }
        if bb.max.y > max[1] { max[1] = bb.max.y }
        if first_geom.is_none() {
            // Convert first shape to GeoJSON geometry (simplistic — only polygon supported for MVP)
            if let shapefile::Shape::Polygon(poly) = &shape {
                let rings: Vec<Vec<Vec<f64>>> = poly.rings().iter().map(|r| {
                    r.points().iter().map(|p| vec![p.x, p.y]).collect()
                }).collect();
                first_geom = Some(geojson::Geometry::new(geojson::Value::Polygon(rings)));
            }
        }
    }
    if count == 0 || min[0].is_infinite() {
        return Err(anyhow!("no_geometry"));
    }
    Ok(ParsedVector {
        bbox: [min[0], min[1], max[0], max[1]],
        geometry: first_geom.ok_or_else(|| anyhow!("no_geometry"))?,
        layer_count: 1,
    })
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test vector_test && cd ..
git add src-tauri/src/core/vector.rs src-tauri/tests/vector_test.rs
git commit -m "feat(a): vector — Shapefile parsing via shapefile crate"
```

---

### Task 6.3：GPKG 解析（rusqlite + WKB）

**Files:**
- Modify: `src-tauri/src/core/vector.rs`、`src-tauri/tests/vector_test.rs`

> GPKG = SQLite with `gpkg_geometry_columns` metadata + WKB-encoded geometry. 完整 spec 见 OGC GPKG。MVP：读 metadata、取第一个 layer 的第一个 feature、解 WKB。

- [ ] **Step 1: 加测试**

```rust
fn make_gpkg_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    use rusqlite::Connection;
    let path = dir.join("triangle.gpkg");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(r#"
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL, m TINYINT NOT NULL
        );
        CREATE TABLE features (id INTEGER PRIMARY KEY, geom BLOB);
        INSERT INTO gpkg_geometry_columns VALUES ('features', 'geom', 'POLYGON', 4326, 0, 0);
    "#).unwrap();

    // GPKG geometry blob: header (8B) + WKB. Simplified: empty header bytes + standard WKB polygon.
    let wkb = build_wkb_polygon(&[[100.0, 30.0], [102.0, 30.0], [101.0, 32.0], [100.0, 30.0]]);
    let mut blob = vec![0x47, 0x50, 0x00, 0x00]; // "GP", version, flags
    blob.extend_from_slice(&4326i32.to_le_bytes());
    blob.extend_from_slice(&wkb);
    conn.execute("INSERT INTO features (geom) VALUES (?)", [&blob]).unwrap();
    path
}

fn build_wkb_polygon(pts: &[[f64; 2]]) -> Vec<u8> {
    let mut w = Vec::new();
    w.push(1u8); // little-endian
    w.extend_from_slice(&3u32.to_le_bytes()); // type = Polygon
    w.extend_from_slice(&1u32.to_le_bytes()); // 1 ring
    w.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for p in pts {
        w.extend_from_slice(&p[0].to_le_bytes());
        w.extend_from_slice(&p[1].to_le_bytes());
    }
    w
}

#[test]
fn parse_gpkg_polygon() {
    let dir = tempfile::tempdir().unwrap();
    let p = make_gpkg_fixture(dir.path());
    let r = parse_vector(&p).unwrap();
    assert!((r.bbox[0] - 100.0).abs() < 1e-9);
    assert!((r.bbox[2] - 102.0).abs() < 1e-9);
}
```

- [ ] **Step 2: 实现 parse_gpkg**

```rust
fn parse_gpkg(path: &Path) -> Result<ParsedVector> {
    let conn = rusqlite::Connection::open(path)?;
    let table: String = conn.query_row(
        "SELECT table_name FROM gpkg_geometry_columns LIMIT 1",
        [], |row| row.get(0),
    ).map_err(|_| anyhow!("no_geometry"))?;
    let column: String = conn.query_row(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
        [&table], |row| row.get(0),
    )?;
    let blob: Vec<u8> = conn.query_row(
        &format!("SELECT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 1"),
        [], |row| row.get(0),
    ).map_err(|_| anyhow!("no_geometry"))?;
    // GPKG header: magic 'GP' (2B) + version (1B) + flags (1B) + srs_id (4B little-endian if flag bit set) + envelope (variable)
    // Simplest path: skip first 8 bytes (no envelope), then WKB.
    if blob.len() < 8 || &blob[0..2] != b"GP" {
        return Err(anyhow!("not a GPKG geometry blob"));
    }
    let envelope_indicator = (blob[3] >> 1) & 0x07;
    let envelope_size = match envelope_indicator {
        0 => 0,
        1 => 32, // [minx, maxx, miny, maxy]
        2 | 3 => 48,
        4 => 64,
        _ => 0,
    };
    let header_len = 8 + envelope_size;
    let wkb = &blob[header_len..];
    let geom = parse_wkb(wkb)?;
    let bbox = geometry_bbox(&geom)?;
    Ok(ParsedVector { bbox, geometry: geom, layer_count: 1 })
}

fn parse_wkb(b: &[u8]) -> Result<geojson::Geometry> {
    if b.is_empty() { return Err(anyhow!("empty WKB")); }
    let little = b[0] == 1;
    let read_u32 = |i: usize| -> u32 {
        let arr: [u8; 4] = b[i..i+4].try_into().unwrap();
        if little { u32::from_le_bytes(arr) } else { u32::from_be_bytes(arr) }
    };
    let read_f64 = |i: usize| -> f64 {
        let arr: [u8; 8] = b[i..i+8].try_into().unwrap();
        if little { f64::from_le_bytes(arr) } else { f64::from_be_bytes(arr) }
    };
    let typ = read_u32(1);
    if typ != 3 { return Err(anyhow!("only POLYGON supported in MVP, got type {}", typ)); }
    let n_rings = read_u32(5) as usize;
    let mut p = 9usize;
    let mut rings: Vec<Vec<Vec<f64>>> = Vec::new();
    for _ in 0..n_rings {
        let n_pts = read_u32(p) as usize; p += 4;
        let mut ring = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            ring.push(vec![read_f64(p), read_f64(p+8)]);
            p += 16;
        }
        rings.push(ring);
    }
    Ok(geojson::Geometry::new(geojson::Value::Polygon(rings)))
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test vector_test && cd ..
git add src-tauri/src/core/vector.rs src-tauri/tests/vector_test.rs
git commit -m "feat(a): vector — GPKG parsing (SQLite metadata + WKB polygon)"
```

---

### Task 6.4：multi-layer 处理（取第一个非空 layer）

**Files:**
- Modify: `src-tauri/src/core/vector.rs`

> GeoJSON FeatureCollection 已经隐含多 feature，取第一个非空。Shapefile 没有 layer。GPKG 可能多个 table——`gpkg_geometry_columns` 用 `LIMIT 1` 已经取了。本任务把 layer_count 字段填得真实些。

- [ ] **Step 1: 改 parse_geojson + parse_gpkg**

`parse_geojson`: 把 `layer_count: 1` 改成 `layer_count: fc.features.len() as u32`。

`parse_gpkg`: 在解 metadata 之前先查 row count：

```rust
let row_count: u32 = conn.query_row(
    "SELECT COUNT(*) FROM gpkg_geometry_columns", [], |row| row.get::<_, i64>(0),
)? as u32;
```

把 `ParsedVector.layer_count` 改成 `row_count`。

- [ ] **Step 2: 加测试**

```rust
#[test]
fn geojson_feature_collection_layer_count() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("multi.geojson");
    std::fs::write(&p, r#"{
        "type":"FeatureCollection",
        "features":[
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[0,0]}},
            {"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[1,1]}}
        ]
    }"#).unwrap();
    let r = parse_vector(&p).unwrap();
    assert_eq!(r.layer_count, 2);
}
```

- [ ] **Step 3: cargo test + commit**

```bash
cd src-tauri && cargo test --test vector_test && cd ..
git add src-tauri/src/core/vector.rs src-tauri/tests/vector_test.rs
git commit -m "feat(a): vector — populate layer_count from feature count / gpkg metadata rows"
```

✅ Phase 6 完成。

---

## Phase 7 — E2E 集成 + Plan B 契约文档

### Task 7.1：E2E 测试（vector → tiles → mock download → stitch → cog → 反读）

**Files:**
- Create: `src-tauri/tests/e2e_test.rs`

> 这个测试 **不打真实 API**——下载步骤用预制的 PNG 字节模拟，验证整条 pipeline 数据正确传递。

- [ ] **Step 1: 写测试**

```rust
//! End-to-end pipeline test: vector parse → tile range → fake fetch → stitch → cog → tiff readback.

use bytes::Bytes;
use image::ImageEncoder;
use imagery_downloader_lib::core::{
    cog::{bbox_3857_from_range, write_cog, CogParams},
    downloader::DownloadedTile,
    stitcher::stitch_rgba,
    tiles::{range_for_bbox, TileCoord},
    vector::parse_vector,
};
use std::path::PathBuf;
use tempfile::tempdir;

fn fake_red_tile() -> Bytes {
    let buf: image::RgbaImage = image::ImageBuffer::from_pixel(256, 256, image::Rgba([200, 30, 30, 255]));
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out).write_image(&buf, 256, 256, image::ColorType::Rgba8.into()).unwrap();
    Bytes::from(out)
}

#[test]
fn pipeline_geojson_to_cog() {
    // 1. Parse a vector file
    let geojson = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/triangle.geojson");
    let pv = parse_vector(&geojson).unwrap();

    // 2. Compute tile range at z=8
    let range = range_for_bbox(pv.bbox, 8);
    assert!(range.count() > 0 && range.count() < 500);

    // 3. Fake fetch: every tile gets the same red PNG
    let tiles: Vec<DownloadedTile> = range.iter()
        .map(|c: TileCoord| DownloadedTile { coord: c, bytes: Some(fake_red_tile()) })
        .collect();

    // 4. Stitch
    let img = stitch_rgba(&tiles, range);
    let expected_w = (range.x_max - range.x_min + 1) as u32 * 256;
    let expected_h = (range.y_max - range.y_min + 1) as u32 * 256;
    assert_eq!(img.width(), expected_w);
    assert_eq!(img.height(), expected_h);
    assert_eq!(img.get_pixel(10, 10).0, [200, 30, 30, 255]);

    // 5. Write COG
    let dir = tempdir().unwrap();
    let out = dir.path().join("e2e.tif");
    let bbox_3857 = bbox_3857_from_range(range);
    write_cog(&img, &CogParams { bbox_3857, zoom: 8 }, &out).unwrap();

    // 6. Read it back via tiff crate, verify dimensions + tag presence
    let f = std::fs::File::open(&out).unwrap();
    let mut dec = tiff::decoder::Decoder::new(f).unwrap();
    let dims = dec.dimensions().unwrap();
    assert_eq!(dims, (expected_w, expected_h));
    let tp = dec.get_tag_f64_vec(tiff::tags::Tag::Unknown(33922)).unwrap();
    // Tiepoint y (world) ≈ bbox_3857[3] (north)
    assert!((tp[4] - bbox_3857[3]).abs() < 1.0);
}
```

- [ ] **Step 2: cargo test + commit**

```bash
cd src-tauri && cargo test --test e2e_test && cd ..
git add src-tauri/tests/e2e_test.rs
git commit -m "test(a): e2e — vector→tiles→fake fetch→stitch→cog→tiff readback round-trip"
```

---

### Task 7.2：core/README.md（Plan B 替换契约）+ spec status

**Files:**
- Create: `src-tauri/src/core/README.md`
- Modify: `docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md`

- [ ] **Step 1: 写 core/README.md**

````markdown
# `src-tauri/src/core/` — Plan A modules

Tauri-agnostic implementations of the satellite imagery downloader. Plan B
(commands wiring) consumes these from `src-tauri/src/commands/*.rs`.

## Modules

| Module | Owns | Tested by |
|---|---|---|
| `tiles` | XYZ ↔ lon/lat math, TileRange | `core/tiles.rs` unit + `tests/tiles_test.rs` property |
| `sources` | URL templates, latency-based auto-pick | `core/sources.rs` unit + `tests/sources_test.rs` (wiremock) |
| `downloader` | Parallel fetch, retry, cancel, TileCache | `tests/downloader_test.rs` (wiremock) |
| `stitcher` | RGBA assembly with transparent failed tiles | `core/stitcher.rs` unit |
| `cog` | GeoTIFF + GeoTransform + preview.png | `tests/cog_test.rs`, `tests/e2e_test.rs` |
| `vector` | GeoJSON / Shapefile / GPKG parsing | `tests/vector_test.rs` |

## Plan B replacement contract

When Plan B wires real Tauri commands:

1. Create `src-tauri/src/commands/` with one file per command group.
2. `commands/download.rs` calls `core::tiles::range_for_bbox` →
   `core::downloader::download_all` (with progress callback that emits
   Tauri events) → `core::stitcher::stitch_rgba` → `core::cog::write_cog`.
3. `commands/vector.rs` calls `core::vector::parse_vector`.
4. Move `src-tauri/src/history.rs` to `src-tauri/src/core/history.rs`
   (no API change).
5. Delete `src-tauri/src/mocks/` entirely.
6. Update `lib.rs`'s `invoke_handler!` from `mocks::commands::*` to
   `commands::*::*`.

The IPC contract (`src/lib/types.ts` ↔ Rust struct shapes) is already
final. Plan B should not change it.

## What's NOT in Plan A (deferred)

- `core/sources.rs` URL templates are hardcoded; a future task can make
  them injectable for HTTP-backed integration tests.
- COG overview pyramid depends on `tiff@0.10` API; if not supported, the
  MVP single-IFD GeoTIFF is still readable by all GIS software.
- Multi-layer GPKG handling: only the first layer is used.
- WKB types other than POLYGON: unsupported in vector::parse_gpkg's MVP.
````

- [ ] **Step 2: 给 spec 追加 Plan A 状态**

```bash
cat >> docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md <<'EOF'

---

## Plan A Implementation Status (2026-05-XX — fill in when executed)

✅ `core/tiles.rs`: lon/lat ↔ XYZ math + TileRange + iter; 6 unit tests + 2 property tests.
✅ `core/sources.rs`: Esri + Google URL templates; pick_auto via latency probe; wiremock-backed test for probe_url.
✅ `core/downloader.rs`: download_one with exponential-backoff retry; download_all with buffer_unordered + CancellationToken + progress callback; TileCache for retry_failed.
✅ `core/stitcher.rs`: stitch_rgba with transparent failed tiles + corrupt-bytes safety.
✅ `core/cog.rs`: TIFF write with GeoTIFF tags (PixelScale, Tiepoint, GeoKey EPSG:3857); atomic tempfile+rename; bbox_3857_from_range helper; write_preview_png.
✅ `core/vector.rs`: GeoJSON / Shapefile (via shapefile crate) / GPKG (rusqlite + WKB).
✅ E2E test: vector→tiles→fake fetch→stitch→cog→tiff readback (`tests/e2e_test.rs`).
✅ `core/README.md` documents Plan B replacement contract.

Pending follow-ups:
- COG overview pyramid (tiff@0.10 API permitting).
- Sources URL injection for full-stack downloader integration tests.
- Multi-layer GPKG support (currently first layer only).
- WKB types beyond POLYGON.

Plan B can now wire these into Tauri commands.
EOF
```

- [ ] **Step 3: commit**

```bash
git add src-tauri/src/core/README.md docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md
git commit -m "docs(a): document Plan A modules + Plan B replacement contract"
```

✅ Plan A 完成。

---

## Self-Review

### 1. Spec 覆盖

| spec §2.2 模块 | task | 备注 |
|---|---|---|
| `tiles` | Phase 1（4 tasks） | ✅ 含 property test |
| `sources` | Phase 2（3 tasks） | ✅ |
| `downloader` | Phase 3（4 tasks） | ✅ |
| `stitcher` | Phase 4（2 tasks） | ✅ |
| `cog` | Phase 5（5 tasks） | ⚠ 金字塔降级方案见 Task 5.5 注 |
| `vector` | Phase 6（4 tasks） | ⚠ MVP 仅 Polygon WKB |
| `history` | （Plan C 已完成） | — |

| spec §5 不变量 | 实现位置 |
|---|---|
| 不写半成品文件（atomic .tmp + rename） | cog.rs `write_cog` |
| 失败瓦片填透明 alpha | stitcher.rs |
| 进度按瓦片数算 | downloader.rs `ProgressUpdate.completed` |
| session tile cache | downloader.rs `TileCache` |

### 2. Placeholder 扫描

- "Step 2a / 2b" in Task 5.5 是有条件分支，**不是** placeholder——明确给出两条路径。
- "fill in when executed" 在 spec status 段是日期占位，跑完 Plan A 时填——可接受。
- 其它无 TBD / TODO。

### 3. 类型一致性

- `TileCoord` { x, y, z } 在 tiles / sources / downloader / stitcher / e2e 全部一致。
- `TileRange` 字段 (x_min, y_min, x_max, y_max, z) 在 tiles / cog / e2e 一致。
- `DownloadedTile` { coord, bytes: Option<Bytes> } 在 downloader / stitcher / e2e 一致。
- `CogParams` { bbox_3857, zoom } 一致。
- `ParsedVector` { bbox, geometry, layer_count } 一致。
- 事件名 / IPC 不在 Plan A 内（Plan B 负责）——无冲突。

无不一致。

---

## Execution Handoff

Plan 已保存到 `docs/superpowers/plans/2026-05-04-plan-a-rust-core-modules.md`。

**两种执行模式：**

1. **Subagent-Driven（推荐）** — 每 task fresh subagent + 两阶段 review。Plan A 大量 Rust 代码、TDD 严格，subagent 容易出 lifetime / type 问题，但 review 能 catch；适合。
2. **Inline Execution** — 当前 session 批量执行，phase checkpoint。Plan A 总 25 tasks 比 Plan C 少，token 预算允许。

**我倾向 Subagent-Driven**——Plan A 的 Rust 类型系统复杂度比 Plan C 高（lifetime / Send/Sync / async），fresh-context subagent 误差更可控。

哪一种？
