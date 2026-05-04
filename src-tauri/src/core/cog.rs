//! Hand-written GeoTIFF / Cloud-Optimized GeoTIFF writer.
//!
//! MVP shape:
//! - Single full-resolution IFD, RGBA8.
//! - GeoTIFF tags (PixelScale + Tiepoint + GeoKey EPSG:3857).
//! - Atomic write via tempfile + rename.
//! - Optional preview.png via the image crate.

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
