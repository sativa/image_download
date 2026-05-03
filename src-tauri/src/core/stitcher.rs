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
        assert_eq!(img.get_pixel(128, 128), &image::Rgba([255, 0, 0, 255]));
        assert_eq!(img.get_pixel(384, 128), &image::Rgba([255, 0, 0, 255]));
    }

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
}
