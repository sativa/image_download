# `src-tauri/src/core/` — Plan A modules

Tauri-agnostic implementations of the satellite imagery downloader. Plan B
(commands wiring) consumes these from `src-tauri/src/commands/*.rs`.

## Modules

| Module | Owns | Tested by |
|---|---|---|
| `tiles` | XYZ ↔ lon/lat math, TileRange | `core/tiles.rs` unit (13) + `tests/tiles_test.rs` property (2) |
| `sources` | URL templates, latency-based auto-pick | `core/sources.rs` unit (3) + `tests/sources_test.rs` wiremock (2) |
| `downloader` | Parallel fetch, retry, cancel, TileCache | `tests/downloader_test.rs` wiremock (6) |
| `stitcher` | RGBA assembly with transparent failed tiles | `core/stitcher.rs` unit (3) |
| `cog` | GeoTIFF + GeoTransform + preview.png | `tests/cog_test.rs` (3), `tests/e2e_test.rs` (1) |
| `vector` | GeoJSON / Shapefile / GPKG parsing | `tests/vector_test.rs` (3) |

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

## Known follow-ups (deferred)

- **COG compression**: tiff@0.10's typed-encoder API for compression is
  awkward; uncompressed RGBA is the MVP. Switching to deflate cuts file
  size 3-5×. Either upgrade tiff or hand-roll the compressed stream.
- **COG overview pyramid**: tiff@0.10's encoder is single-IFD only.
  Adding a pyramid requires multi-IFD writes — wait for tiff upgrade or
  hand-roll TIFF byte layout.
- **Sources URL injection**: URLs hardcoded; full-stack downloader
  integration tests blocked until the URL template is overridable.
- **Multi-layer GPKG**: only the first layer is parsed.
- **WKB types beyond POLYGON**: unsupported in vector::parse_gpkg's MVP.
- **Shapefile fixture in unit tests**: parser code present but no
  generative test; needs `shapefile::Writer` setup.
