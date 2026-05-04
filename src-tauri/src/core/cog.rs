//! Hand-written GeoTIFF / Cloud-Optimized GeoTIFF writer.
//!
//! MVP shape:
//! - Single full-resolution IFD, RGBA8.
//! - GeoTIFF tags (PixelScale + Tiepoint + GeoKey EPSG:3857).
//! - Atomic write via tempfile + rename.
//! - Optional preview.png via the image crate.
//!
//! Overview pyramid not implemented: tiff@0.10's encoder API only allows one
//! image per file via `new_image()`. Adding pyramid requires either upgrading
//! to a future tiff release that supports multi-IFD writes, or hand-rolling the
//! TIFF byte layout (multiple IFDs chained via NextIFDOffset). Files written
//! here are still valid GeoTIFFs that QGIS / GDAL load correctly — just not
//! "cloud-optimized" in the streaming sense. Plan-B follow-up.

use anyhow::Result;
use image::RgbaImage;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tiff::encoder::{colortype, TiffEncoder};
use tiff::tags::Tag;

const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;

#[derive(Debug, Clone)]
pub struct CogParams {
    /// [west, south, east, north] in EPSG:3857 meters.
    pub bbox_3857: [f64; 4],
    pub zoom: u32,
}

pub fn write_cog(img: &RgbaImage, p: &CogParams, path: &Path) -> Result<()> {
    let tmp = path.with_extension("tif.tmp");
    {
        let f = File::create(&tmp)?;
        let mut enc = TiffEncoder::new(BufWriter::new(f))?;
        let mut tiff_img = enc.new_image::<colortype::RGBA8>(img.width(), img.height())?;

        let pixel_size_x = (p.bbox_3857[2] - p.bbox_3857[0]) / img.width() as f64;
        let pixel_size_y = (p.bbox_3857[3] - p.bbox_3857[1]) / img.height() as f64;
        let pixel_scale: [f64; 3] = [pixel_size_x, pixel_size_y, 0.0];
        tiff_img
            .encoder()
            .write_tag(Tag::Unknown(TAG_MODEL_PIXEL_SCALE), &pixel_scale[..])?;

        // Tiepoint: image (0,0,0) → world (west, north, 0).
        let tiepoint: [f64; 6] = [0.0, 0.0, 0.0, p.bbox_3857[0], p.bbox_3857[3], 0.0];
        tiff_img
            .encoder()
            .write_tag(Tag::Unknown(TAG_MODEL_TIEPOINT), &tiepoint[..])?;

        // GeoKey Directory: declare CRS = EPSG:3857.
        // Header (4 u16): KeyDirectoryVersion=1, KeyRevision=1, MinorRevision=1, NumberOfKeys=3.
        // Then 3 quadruples (KeyID, TIFFTagLocation=0, Count=1, Value).
        let geokeys: [u16; 4 + 4 * 3] = [
            1, 1, 1, 3,
            1024, 0, 1, 1,    // GTModelTypeGeoKey = ModelTypeProjected
            1025, 0, 1, 1,    // GTRasterTypeGeoKey = RasterPixelIsArea
            3072, 0, 1, 3857, // ProjectedCSTypeGeoKey = EPSG:3857
        ];
        tiff_img
            .encoder()
            .write_tag(Tag::Unknown(TAG_GEO_KEY_DIRECTORY), &geokeys[..])?;

        tiff_img.write_data(img.as_raw())?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

use crate::core::tiles::TileRange;

const EARTH_HALF_CIRC_M: f64 = 20037508.3427892;

/// Convert a TileRange to its EPSG:3857 bbox [west, south, east, north].
pub fn bbox_3857_from_range(r: TileRange) -> [f64; 4] {
    let n = 2_f64.powi(r.z as i32);
    let cell = 2.0 * EARTH_HALF_CIRC_M / n;
    let west = -EARTH_HALF_CIRC_M + r.x_min as f64 * cell;
    let east = -EARTH_HALF_CIRC_M + (r.x_max + 1) as f64 * cell;
    let north = EARTH_HALF_CIRC_M - r.y_min as f64 * cell;
    let south = EARTH_HALF_CIRC_M - (r.y_max + 1) as f64 * cell;
    [west, south, east, north]
}

/// Write a downsampled PNG preview alongside the GeoTIFF.
/// `max_dim` caps the longer edge — small previews load instantly in image viewers.
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
    preview.save_with_format(&tmp, image::ImageFormat::Png)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tiles::TileRange;

    #[test]
    fn world_at_zoom_zero() {
        let bb = bbox_3857_from_range(TileRange { x_min: 0, y_min: 0, x_max: 0, y_max: 0, z: 0 });
        assert!((bb[0] + EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[2] - EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[3] - EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[1] + EARTH_HALF_CIRC_M).abs() < 1.0);
    }
}

