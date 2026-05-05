//! Tile stitcher: assemble downloaded JPEG/PNG tiles into a single RGBA image.

use crate::core::downloader::DownloadedTile;
use crate::core::tiles::TileRange;
use image::{ImageBuffer, Rgba, RgbaImage};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TILE_PX: u32 = 256;

/// Stitch tiles into a single RgbaImage covering `range`. Failed tiles (bytes=None)
/// or undecodable bytes leave their region transparent (alpha = 0).
///
/// Decode runs in parallel via rayon (the CPU-bound part); composite is serial
/// (memory-bound, fast even single-threaded). `on_progress(done, total)` is
/// invoked at ~4 Hz from the decode pool so callers can emit progress events.
pub fn stitch_rgba_with_progress<F>(
    tiles: &[DownloadedTile],
    range: TileRange,
    on_progress: F,
) -> RgbaImage
where
    F: Fn(u32, u32) + Send + Sync,
{
    let tx = (range.x_max - range.x_min + 1) as u32;
    let ty = (range.y_max - range.y_min + 1) as u32;
    let total = tiles.len() as u32;

    let counter = Arc::new(AtomicU32::new(0));
    let last_emit = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1)));
    let on_progress = Arc::new(on_progress);

    // Phase 1: decode + position lookup in parallel.
    let decoded: Vec<(u32, u32, RgbaImage)> = tiles
        .par_iter()
        .filter_map(|tile| {
            let result = decode_tile(tile, range);
            // Always increment — counter tracks "tiles processed" not "tiles successful";
            // matches the user-facing "N / total" expectation.
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            let mut last = last_emit.lock().unwrap();
            let now = Instant::now();
            let force = done == total;
            if force || now.duration_since(*last) >= Duration::from_millis(250) {
                *last = now;
                drop(last);
                on_progress(done, total);
            }
            result
        })
        .collect();

    // Phase 2: composite serially. ImageBuffer's underlying Vec<u8> is one big
    // allocation we mutate in place; not worth parallelizing the writes.
    let mut img: RgbaImage =
        ImageBuffer::from_pixel(tx * TILE_PX, ty * TILE_PX, Rgba([0, 0, 0, 0]));
    for (dx, dy, rgba) in decoded {
        image::imageops::replace(&mut img, &rgba, dx as i64, dy as i64);
    }
    on_progress(total, total);
    img
}

fn decode_tile(tile: &DownloadedTile, range: TileRange) -> Option<(u32, u32, RgbaImage)> {
    let bytes = tile.bytes.as_ref()?;
    let dyn_img = image::load_from_memory(bytes).ok()?;
    let rgba = dyn_img.to_rgba8();
    if rgba.width() != TILE_PX || rgba.height() != TILE_PX {
        return None;
    }
    let dx = ((tile.coord.x - range.x_min) as u32) * TILE_PX;
    let dy = ((tile.coord.y - range.y_min) as u32) * TILE_PX;
    Some((dx, dy, rgba))
}

/// Convenience wrapper — same behaviour as the old single-threaded API but now
/// gets the rayon parallelism for free. Use `stitch_rgba_with_progress` when
/// you want to emit progress events.
pub fn stitch_rgba(tiles: &[DownloadedTile], range: TileRange) -> RgbaImage {
    stitch_rgba_with_progress(tiles, range, |_, _| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::downloader::DownloadedTile;
    use crate::core::tiles::{TileCoord, TileRange};
    use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};

    fn red_png_tile() -> bytes::Bytes {
        let buf: image::RgbaImage =
            image::ImageBuffer::from_pixel(256, 256, image::Rgba([255, 0, 0, 255]));
        let mut out = Vec::new();
        PngEncoder::new(&mut out)
            .write_image(&buf, 256, 256, ColorType::Rgba8.into())
            .unwrap();
        bytes::Bytes::from(out)
    }

    #[test]
    fn stitch_2x1_red_tiles() {
        let range = TileRange {
            x_min: 0,
            y_min: 0,
            x_max: 1,
            y_max: 0,
            z: 5,
        };
        let tiles = vec![
            DownloadedTile {
                coord: TileCoord { x: 0, y: 0, z: 5 },
                bytes: Some(red_png_tile()),
            },
            DownloadedTile {
                coord: TileCoord { x: 1, y: 0, z: 5 },
                bytes: Some(red_png_tile()),
            },
        ];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 256);
        assert_eq!(img.get_pixel(128, 128), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(384, 128), &image::Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn missing_tile_leaves_transparent_region() {
        let range = TileRange {
            x_min: 0,
            y_min: 0,
            x_max: 1,
            y_max: 0,
            z: 5,
        };
        let tiles = vec![
            DownloadedTile {
                coord: TileCoord { x: 0, y: 0, z: 5 },
                bytes: Some(red_png_tile()),
            },
            DownloadedTile {
                coord: TileCoord { x: 1, y: 0, z: 5 },
                bytes: None,
            },
        ];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.get_pixel(128, 128).0[3], 255, "tile 0 should be opaque");
        assert_eq!(
            img.get_pixel(384, 128).0[3],
            0,
            "tile 1 should be transparent (failed)"
        );
    }

    #[test]
    fn corrupt_bytes_treated_as_failed() {
        let range = TileRange {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
            z: 5,
        };
        let tiles = vec![DownloadedTile {
            coord: TileCoord { x: 0, y: 0, z: 5 },
            bytes: Some(bytes::Bytes::from_static(b"not an image")),
        }];
        let img = stitch_rgba(&tiles, range);
        assert_eq!(img.get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn progress_callback_reaches_total() {
        let range = TileRange {
            x_min: 0,
            y_min: 0,
            x_max: 1,
            y_max: 0,
            z: 5,
        };
        let tiles = vec![
            DownloadedTile {
                coord: TileCoord { x: 0, y: 0, z: 5 },
                bytes: Some(red_png_tile()),
            },
            DownloadedTile {
                coord: TileCoord { x: 1, y: 0, z: 5 },
                bytes: Some(red_png_tile()),
            },
        ];
        let last_done = Arc::new(AtomicU32::new(0));
        let last_total = Arc::new(AtomicU32::new(0));
        let d2 = last_done.clone();
        let t2 = last_total.clone();
        let _ = stitch_rgba_with_progress(&tiles, range, move |done, total| {
            d2.store(done, Ordering::Relaxed);
            t2.store(total, Ordering::Relaxed);
        });
        assert_eq!(last_done.load(Ordering::Relaxed), 2);
        assert_eq!(last_total.load(Ordering::Relaxed), 2);
    }
}
