//! Hand-written GeoTIFF / Cloud-Optimized GeoTIFF writer.
//!
//! Three output compressions, selected by [`Compression`]:
//! - `Jpeg`    — lossy YCbCr JPEG, 3-band RGB (alpha dropped). ~10x smaller; the
//!   form GDAL/QGIS and major tile providers use for RGB imagery (our source
//!   tiles are already JPEG, so re-encoding at high quality is near-transparent).
//!   tiff@0.10's encoder cannot emit JPEG, so this path hand-rolls the TIFF byte
//!   layout: one strip whose data is an abbreviated JPEG datastream, with the
//!   quant/Huffman tables in JPEGTables (347) — exactly GDAL's structure
//!   (PHOTOMETRIC=YCBCR + RefBlackWhite + YCbCrSubSampling).
//! - `Deflate` — lossless DEFLATE + horizontal predictor, 3-band RGB. ~2x.
//! - `None`    — uncompressed RGBA8 (4-band). Largest; exact legacy output.
//!
//! All paths write GeoTIFF tags (PixelScale + Tiepoint + GeoKey EPSG:3857) and
//! commit atomically via tempfile + rename. Overview pyramids are not written
//! (single full-res IFD); files load correctly in QGIS/GDAL but aren't
//! "cloud-optimized" in the streaming sense.

use anyhow::{anyhow, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, RgbaImage};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tiff::encoder::{colortype, Compression as TiffCompression, DeflateLevel, Predictor, TiffEncoder};
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

/// Output compression for the written GeoTIFF.
#[derive(Debug, Clone, Copy)]
pub enum Compression {
    /// Lossy YCbCr JPEG, 3-band RGB (alpha dropped). ~10x smaller. Default.
    Jpeg { quality: u8 },
    /// Lossless DEFLATE + horizontal predictor, 3-band RGB (alpha dropped). ~2x.
    Deflate,
    /// Uncompressed RGBA8 (4-band, alpha kept). Exact legacy output.
    None,
}

impl Default for Compression {
    fn default() -> Self {
        Compression::Jpeg { quality: 95 }
    }
}

pub fn write_cog(img: &RgbaImage, p: &CogParams, compression: Compression, path: &Path) -> Result<()> {
    match compression {
        Compression::Jpeg { quality } => {
            // libjpeg caps each dimension at 65535 px; for anything larger (huge
            // GUI selections) fall back to lossless DEFLATE rather than fail.
            if img.width() > 65_500 || img.height() > 65_500 {
                write_tiffcrate(img, p, true, TiffCompression::Deflate(DeflateLevel::Best), Predictor::Horizontal, path)
            } else {
                write_jpeg_geotiff(img, p, quality, path)
            }
        }
        Compression::Deflate => write_tiffcrate(
            img, p, true, TiffCompression::Deflate(DeflateLevel::Best), Predictor::Horizontal, path,
        ),
        Compression::None => {
            write_tiffcrate(img, p, false, TiffCompression::Uncompressed, Predictor::None, path)
        }
    }
}

// ---- pure-tiff-crate paths (lossless): None = RGBA8, Deflate = RGB8 ----

fn write_tiffcrate(
    img: &RgbaImage,
    p: &CogParams,
    drop_alpha: bool,
    compression: TiffCompression,
    predictor: Predictor,
    path: &Path,
) -> Result<()> {
    let tmp = path.with_extension("tif.tmp");
    // Classic TIFF caps file size at 4 GB (32-bit offsets). Switch to BigTIFF when
    // the raw payload would breach it (compression only shrinks, so this is safe).
    let channels: u64 = if drop_alpha { 3 } else { 4 };
    let needs_bigtiff = img.width() as u64 * img.height() as u64 * channels > 3_500_000_000;
    {
        let f = File::create(&tmp)?;
        let buf = BufWriter::new(f);
        if needs_bigtiff {
            let mut enc = TiffEncoder::new_big(buf)?
                .with_compression(compression)
                .with_predictor(predictor);
            write_geo_pixels(&mut enc, img, p, drop_alpha)?;
        } else {
            let mut enc = TiffEncoder::new(buf)?
                .with_compression(compression)
                .with_predictor(predictor);
            write_geo_pixels(&mut enc, img, p, drop_alpha)?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn write_geo_pixels<W, K>(
    enc: &mut TiffEncoder<W, K>,
    img: &RgbaImage,
    p: &CogParams,
    drop_alpha: bool,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    K: tiff::encoder::TiffKind,
{
    let (w, h) = (img.width(), img.height());
    if drop_alpha {
        let mut tiff_img = enc.new_image::<colortype::RGB8>(w, h)?;
        write_geo_tags(tiff_img.encoder(), w, h, p)?;
        let raw = img.as_raw();
        let mut rgb = Vec::with_capacity(raw.len() / 4 * 3);
        for px in raw.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        tiff_img.write_data(&rgb)?;
    } else {
        let mut tiff_img = enc.new_image::<colortype::RGBA8>(w, h)?;
        write_geo_tags(tiff_img.encoder(), w, h, p)?;
        tiff_img.write_data(img.as_raw())?;
    }
    Ok(())
}

fn write_geo_tags<W, K>(
    dir: &mut tiff::encoder::DirectoryEncoder<'_, W, K>,
    w: u32,
    h: u32,
    p: &CogParams,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    K: tiff::encoder::TiffKind,
{
    let pixel_scale: [f64; 3] = [
        (p.bbox_3857[2] - p.bbox_3857[0]) / w as f64,
        (p.bbox_3857[3] - p.bbox_3857[1]) / h as f64,
        0.0,
    ];
    dir.write_tag(Tag::Unknown(TAG_MODEL_PIXEL_SCALE), &pixel_scale[..])?;
    let tiepoint: [f64; 6] = [0.0, 0.0, 0.0, p.bbox_3857[0], p.bbox_3857[3], 0.0];
    dir.write_tag(Tag::Unknown(TAG_MODEL_TIEPOINT), &tiepoint[..])?;
    let geokeys: [u16; 16] = [
        1, 1, 1, 3, 1024, 0, 1, 1, // GTModelType = Projected
        1025, 0, 1, 1, // GTRasterType = PixelIsArea
        3072, 0, 1, 3857, // ProjectedCSType = EPSG:3857
    ];
    dir.write_tag(Tag::Unknown(TAG_GEO_KEY_DIRECTORY), &geokeys[..])?;
    Ok(())
}

// ---- hand-rolled single-file YCbCr JPEG-GeoTIFF (tiff crate can't emit JPEG) ----

fn write_jpeg_geotiff(img: &RgbaImage, p: &CogParams, quality: u8, path: &Path) -> Result<()> {
    let (w, h) = (img.width(), img.height());
    // RGB bytes (drop the all-opaque alpha — JPEG is 3-band).
    let raw = img.as_raw();
    let mut rgb = Vec::with_capacity(raw.len() / 4 * 3);
    for px in raw.chunks_exact(4) {
        rgb.extend_from_slice(&px[..3]);
    }
    // Encode a standard YCbCr JFIF (the image crate uses 4:2:2 subsampling).
    let mut jfif: Vec<u8> = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut jfif, quality);
        enc.encode(&rgb, w, h, ExtendedColorType::Rgb8)?;
    }
    let (tables, strip, hsamp, vsamp) = split_jfif(&jfif)?;
    let bytes = build_jpeg_tiff(&tables, &strip, w, h, hsamp, vsamp, p);
    let tmp = path.with_extension("tif.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Split a baseline JFIF into a libtiff-style abbreviated pair:
/// `(jpegtables = SOI+DQT+DHT+EOI, strip = SOI+SOF+SOS..EOI, h_subsample, v_subsample)`.
fn split_jfif(jfif: &[u8]) -> Result<(Vec<u8>, Vec<u8>, u8, u8)> {
    let n = jfif.len();
    let mut i = 2; // skip SOI (FFD8)
    let mut dqt: Vec<u8> = Vec::new();
    let mut dht: Vec<u8> = Vec::new();
    let mut sof: Option<&[u8]> = None;
    let mut sos_pos: Option<usize> = None;
    while i + 1 < n {
        if jfif[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = jfif[i + 1];
        if marker == 0xD9 {
            break; // EOI
        }
        if marker == 0xDA {
            sos_pos = Some(i); // SOS — entropy data runs to EOI
            break;
        }
        let seg_len = ((jfif[i + 2] as usize) << 8) | jfif[i + 3] as usize;
        let seg = &jfif[i..i + 2 + seg_len];
        match marker {
            0xDB => dqt.extend_from_slice(seg),                 // DQT
            0xC4 => dht.extend_from_slice(seg),                 // DHT
            0xC0 | 0xC1 | 0xC2 => sof = Some(seg),              // SOF0/1/2
            _ => {}                                             // APPn etc. dropped
        }
        i += 2 + seg_len;
    }
    let sof = sof.ok_or_else(|| anyhow!("JPEG has no SOF marker"))?;
    let sos_pos = sos_pos.ok_or_else(|| anyhow!("JPEG has no SOS marker"))?;
    // Component-0 (Y) sampling factors: SOF byte layout is
    // [FFCx, len(2), precision, H(2), W(2), Ncomp, id, HiVi, qtable, ...].
    let sampling = sof[11];
    let (hsamp, vsamp) = (sampling >> 4, sampling & 0x0F);
    let mut tables = vec![0xFF, 0xD8];
    tables.extend_from_slice(&dqt);
    tables.extend_from_slice(&dht);
    tables.extend_from_slice(&[0xFF, 0xD9]);
    let mut strip = vec![0xFF, 0xD8];
    strip.extend_from_slice(sof);
    strip.extend_from_slice(&jfif[sos_pos..]); // SOS header + entropy + EOI
    Ok((tables, strip, hsamp, vsamp))
}

/// Assemble a classic little-endian GeoTIFF whose single strip is the abbreviated
/// JPEG, mirroring GDAL's `COMPRESS=JPEG PHOTOMETRIC=YCBCR` byte layout.
fn build_jpeg_tiff(tables: &[u8], strip: &[u8], w: u32, h: u32, hs: u8, vs: u8, p: &CogParams) -> Vec<u8> {
    const SHORT: u16 = 3;
    const LONG: u16 = 4;
    const RATIONAL: u16 = 5;
    const UNDEFINED: u16 = 7;
    const DOUBLE: u16 = 12;

    let n_tags: u32 = 17;
    let ifd_off: u32 = 8;
    let mut cur = ifd_off + 2 + n_tags * 12 + 4; // first out-of-line offset
    let bps_off = cur;
    cur += 6; // BitsPerSample [8,8,8]
    let sf_off = cur;
    cur += 6; // SampleFormat [1,1,1]
    let rbw_off = cur;
    cur += 48; // RefBlackWhite (6 rationals)
    let ps_off = cur;
    cur += 24; // ModelPixelScale (3 doubles)
    let tp_off = cur;
    cur += 48; // ModelTiepoint (6 doubles)
    let gk_off = cur;
    cur += 32; // GeoKeyDirectory (16 shorts)
    let tab_off = cur;
    cur += tables.len() as u32;
    let strip_off = cur;

    let mut o: Vec<u8> = Vec::with_capacity(strip_off as usize + strip.len());
    o.extend_from_slice(b"II");
    o.extend_from_slice(&42u16.to_le_bytes());
    o.extend_from_slice(&ifd_off.to_le_bytes());
    o.extend_from_slice(&(n_tags as u16).to_le_bytes());

    // 12-byte IFD entry; non-capturing so `o` stays freely usable afterwards.
    fn ifd_entry(o: &mut Vec<u8>, tag: u16, typ: u16, count: u32, valoff: [u8; 4]) {
        o.extend_from_slice(&tag.to_le_bytes());
        o.extend_from_slice(&typ.to_le_bytes());
        o.extend_from_slice(&count.to_le_bytes());
        o.extend_from_slice(&valoff);
    }
    // value field for a SHORT/count-1 tag (low 2 bytes hold the value)
    fn short1(x: u16) -> [u8; 4] {
        let mut b = [0u8; 4];
        b[..2].copy_from_slice(&x.to_le_bytes());
        b
    }
    ifd_entry(&mut o, 256, LONG, 1, w.to_le_bytes()); // ImageWidth
    ifd_entry(&mut o, 257, LONG, 1, h.to_le_bytes()); // ImageLength
    ifd_entry(&mut o, 258, SHORT, 3, bps_off.to_le_bytes()); // BitsPerSample
    ifd_entry(&mut o, 259, SHORT, 1, short1(7)); // Compression = JPEG
    ifd_entry(&mut o, 262, SHORT, 1, short1(6)); // Photometric = YCbCr
    ifd_entry(&mut o, 273, LONG, 1, strip_off.to_le_bytes()); // StripOffsets
    ifd_entry(&mut o, 277, SHORT, 1, short1(3)); // SamplesPerPixel
    ifd_entry(&mut o, 278, LONG, 1, h.to_le_bytes()); // RowsPerStrip = full image
    ifd_entry(&mut o, 279, LONG, 1, (strip.len() as u32).to_le_bytes()); // StripByteCounts
    ifd_entry(&mut o, 284, SHORT, 1, short1(1)); // PlanarConfiguration
    ifd_entry(&mut o, 339, SHORT, 3, sf_off.to_le_bytes()); // SampleFormat
    ifd_entry(&mut o, 347, UNDEFINED, tables.len() as u32, tab_off.to_le_bytes()); // JPEGTables
    {
        // YCbCrSubSampling: two SHORTs fit inline in the 4-byte value field.
        let mut b = [0u8; 4];
        b[..2].copy_from_slice(&(hs as u16).to_le_bytes());
        b[2..].copy_from_slice(&(vs as u16).to_le_bytes());
        ifd_entry(&mut o, 530, SHORT, 2, b);
    }
    ifd_entry(&mut o, 532, RATIONAL, 6, rbw_off.to_le_bytes()); // ReferenceBlackWhite
    ifd_entry(&mut o, 33550, DOUBLE, 3, ps_off.to_le_bytes()); // ModelPixelScale
    ifd_entry(&mut o, 33922, DOUBLE, 6, tp_off.to_le_bytes()); // ModelTiepoint
    ifd_entry(&mut o, 34735, SHORT, 16, gk_off.to_le_bytes()); // GeoKeyDirectory
    o.extend_from_slice(&0u32.to_le_bytes()); // next IFD = none

    // out-of-line blobs, in the offset order assigned above
    for _ in 0..3 {
        o.extend_from_slice(&8u16.to_le_bytes()); // BitsPerSample
    }
    for _ in 0..3 {
        o.extend_from_slice(&1u16.to_le_bytes()); // SampleFormat = unsigned int
    }
    for &(num, den) in &[(0u32, 1u32), (255, 1), (128, 1), (255, 1), (128, 1), (255, 1)] {
        o.extend_from_slice(&num.to_le_bytes());
        o.extend_from_slice(&den.to_le_bytes()); // ReferenceBlackWhite (YCbCr)
    }
    let px_x = (p.bbox_3857[2] - p.bbox_3857[0]) / w as f64;
    let px_y = (p.bbox_3857[3] - p.bbox_3857[1]) / h as f64;
    for v in [px_x, px_y, 0.0] {
        o.extend_from_slice(&v.to_le_bytes()); // ModelPixelScale
    }
    for v in [0.0, 0.0, 0.0, p.bbox_3857[0], p.bbox_3857[3], 0.0] {
        o.extend_from_slice(&v.to_le_bytes()); // ModelTiepoint
    }
    for v in [1u16, 1, 1, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857] {
        o.extend_from_slice(&v.to_le_bytes()); // GeoKeyDirectory -> EPSG:3857
    }
    o.extend_from_slice(tables);
    o.extend_from_slice(strip);
    o
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

/// WGS84 lon/lat → EPSG:3857 metres.
fn lonlat_to_3857(lon: f64, lat: f64) -> (f64, f64) {
    // Same constants the cog writer uses for inverse transforms.
    let r = 6378137.0_f64;
    // Clamp to the Web-Mercator extent so atanh doesn't blow up at the poles.
    let lat = lat.clamp(-85.05112878, 85.05112878);
    let x = lon.to_radians() * r;
    let y = (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0).tan().ln() * r;
    (x, y)
}

/// Crop a stitched RGBA image to the user's WGS84 bbox.
///
/// XYZ downloads always cover whole tiles, so the stitched output spans
/// the tile-snapped extent `outer_3857` (returned by
/// [`bbox_3857_from_range`]). The user's actual bbox is typically a
/// strict subset — at z17 each tile is ~610 m, so a 100 m-wide selection
/// can balloon to >1 km on each side without cropping.
///
/// Returns `(cropped, user_bbox_3857)`. When the user's bbox already
/// covers the entire outer extent (or extends past it), the original
/// image is returned unchanged.
///
/// Why crop at the pixel boundary (round outwards) instead of resampling:
///   * Resampling at sub-pixel offsets blurs the imagery for no real
///     gain — geographic accuracy is anchored to whole pixels in 3857.
///   * Round-outwards keeps every pixel the user might consider "inside"
///     the bbox, never trimming actual content.
pub fn crop_to_user_bbox(
    img: image::RgbaImage,
    outer_3857: [f64; 4],
    user_wgs84: [f64; 4],
) -> (image::RgbaImage, [f64; 4]) {
    let (w_in, h_in) = (img.width(), img.height());
    if w_in == 0 || h_in == 0 {
        return (img, outer_3857);
    }
    let (uw_x, us_y) = lonlat_to_3857(user_wgs84[0], user_wgs84[1]);
    let (ue_x, un_y) = lonlat_to_3857(user_wgs84[2], user_wgs84[3]);

    let (ow_x, os_y, oe_x, on_y) = (outer_3857[0], outer_3857[1], outer_3857[2], outer_3857[3]);
    let px_x = (oe_x - ow_x) / w_in as f64;
    let px_y = (on_y - os_y) / h_in as f64;
    if px_x <= 0.0 || px_y <= 0.0 {
        return (img, outer_3857);
    }

    // Pixel window aligned to the outer extent. Round outwards so we
    // never trim a pixel the user might want; clamp to image bounds.
    let col_min = ((uw_x - ow_x) / px_x).floor() as i64;
    let col_max = ((ue_x - ow_x) / px_x).ceil() as i64; // exclusive
    let row_min = ((on_y - un_y) / px_y).floor() as i64; // y increases southward in image
    let row_max = ((on_y - us_y) / px_y).ceil() as i64;

    let cx0 = col_min.clamp(0, w_in as i64) as u32;
    let cx1 = col_max.clamp(0, w_in as i64) as u32;
    let ry0 = row_min.clamp(0, h_in as i64) as u32;
    let ry1 = row_max.clamp(0, h_in as i64) as u32;
    if cx1 <= cx0 || ry1 <= ry0 {
        // Degenerate — user bbox lies entirely outside the snapped range.
        return (img, outer_3857);
    }
    let crop_w = cx1 - cx0;
    let crop_h = ry1 - ry0;
    // No-op when the crop window covers the whole image already.
    if crop_w == w_in && crop_h == h_in {
        return (img, outer_3857);
    }

    // Compute the exact 3857 bbox of the chosen pixel window so the
    // GeoTIFF tags match what we actually wrote.
    let new_west = ow_x + cx0 as f64 * px_x;
    let new_east = ow_x + cx1 as f64 * px_x;
    let new_north = on_y - ry0 as f64 * px_y;
    let new_south = on_y - ry1 as f64 * px_y;

    let cropped = image::imageops::crop_imm(&img, cx0, ry0, crop_w, crop_h).to_image();
    (cropped, [new_west, new_south, new_east, new_north])
}

/// Write a downsampled PNG preview alongside the GeoTIFF.
/// `max_dim` caps the longer edge — small previews load instantly in image viewers.
///
/// Uses **parallel nearest-neighbor sampling**: O(out_w × out_h) reads
/// independent of source size, parallelised across rows via rayon. A
/// single-threaded Triangle filter (the previous implementation) takes ~10
/// minutes on a 71,680 × 81,920 RGBA8 source — this version finishes in
/// well under a second on the same input. The trade-off is reduced
/// anti-aliasing, which is acceptable for a "preview" sidecar.
pub fn write_preview_png(img: &RgbaImage, path: &Path, max_dim: u32) -> Result<()> {
    let src_w = img.width();
    let src_h = img.height();
    let max_src = src_w.max(src_h);

    // Source already ≤ max_dim: no downsample, just save.
    if max_src <= max_dim {
        let tmp = path.with_extension("png.tmp");
        img.save_with_format(&tmp, image::ImageFormat::Png)?;
        std::fs::rename(&tmp, path)?;
        return Ok(());
    }

    let scale = max_dim as f64 / max_src as f64;
    let out_w = ((src_w as f64 * scale).round() as u32).max(1);
    let out_h = ((src_h as f64 * scale).round() as u32).max(1);

    // Direct raw-byte access bypasses ImageBuffer::get_pixel's per-call bounds
    // check; rayon par_chunks_mut writes each output row in parallel — rows
    // are independent so no synchronization needed.
    let raw = img.as_raw();
    let src_w_usize = src_w as usize;
    let stride = (out_w * 4) as usize;
    let mut buf = vec![0u8; (out_w as usize) * (out_h as usize) * 4];

    use rayon::prelude::*;
    buf.par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            // Use u64 in the index math to stay safe at >2^32 source pixels
            // (a 5.9 GB source has 1.5 G pixels — well above u32::MAX).
            let src_y = ((y as u64 * src_h as u64) / out_h as u64) as usize;
            let row_byte_off = src_y * src_w_usize * 4;
            for x in 0..(out_w as usize) {
                let src_x = ((x as u64 * src_w as u64) / out_w as u64) as usize;
                let off = row_byte_off + src_x * 4;
                row[x * 4..x * 4 + 4].copy_from_slice(&raw[off..off + 4]);
            }
        });

    let preview = RgbaImage::from_raw(out_w, out_h, buf)
        .ok_or_else(|| anyhow::anyhow!("preview buffer size mismatch"))?;
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
        let bb = bbox_3857_from_range(TileRange {
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
            z: 0,
        });
        assert!((bb[0] + EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[2] - EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[3] - EARTH_HALF_CIRC_M).abs() < 1.0);
        assert!((bb[1] + EARTH_HALF_CIRC_M).abs() < 1.0);
    }

    fn tiny_params() -> CogParams {
        CogParams {
            bbox_3857: [11_561_289.23, 4_298_125.76, 11_563_516.65, 4_300_878.69],
            zoom: 17,
        }
    }

    /// The hand-rolled JPEG GeoTIFF must be a well-formed little-endian TIFF whose
    /// IFD declares JPEG compression (259=7) + YCbCr photometric (262=6) — the
    /// structure GDAL/QGIS read. (Pixel decode is exercised end-to-end via rasterio
    /// in the CLI integration test; the `tiff` crate's decoder can't decompress JPEG.)
    #[test]
    fn jpeg_geotiff_has_expected_tiff_structure() {
        let img = image::RgbaImage::from_fn(64, 48, |x, y| {
            image::Rgba([x as u8 * 3, y as u8 * 5, 100, 255])
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jpeg.tif");
        write_cog(&img, &tiny_params(), Compression::Jpeg { quality: 90 }, &path).unwrap();
        let b = std::fs::read(&path).unwrap();

        assert_eq!(&b[0..2], b"II", "little-endian byte order mark");
        assert_eq!(u16::from_le_bytes([b[2], b[3]]), 42, "classic TIFF magic");
        let ifd = u32::from_le_bytes([b[4], b[5], b[6], b[7]]) as usize;
        let n = u16::from_le_bytes([b[ifd], b[ifd + 1]]) as usize;
        assert_eq!(n, 17, "tag count");

        let mut tags = std::collections::HashMap::new();
        for i in 0..n {
            let e = ifd + 2 + i * 12;
            let tag = u16::from_le_bytes([b[e], b[e + 1]]);
            let val = u16::from_le_bytes([b[e + 8], b[e + 9]]);
            tags.insert(tag, val);
        }
        assert_eq!(tags.get(&259), Some(&7), "Compression = JPEG");
        assert_eq!(tags.get(&262), Some(&6), "Photometric = YCbCr");
        assert_eq!(tags.get(&277), Some(&3), "SamplesPerPixel = 3 (alpha dropped)");
        assert!(tags.contains_key(&347), "JPEGTables present");
        assert!(tags.contains_key(&34735), "GeoKeyDirectory present");
    }

    /// Lossless paths round-trip through the `tiff` crate decoder with the right
    /// band count: None = 4 (RGBA, legacy), Deflate = 3 (RGB, alpha dropped).
    #[test]
    fn lossless_paths_band_counts() {
        let img = image::RgbaImage::from_fn(40, 32, |x, _| image::Rgba([x as u8, 7, 9, 255]));
        for (mode, want_bands) in [(Compression::None, 4u16), (Compression::Deflate, 3)] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("loss.tif");
            write_cog(&img, &tiny_params(), mode, &path).unwrap();
            let f = std::fs::File::open(&path).unwrap();
            let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).unwrap();
            let spp = dec
                .get_tag_u32(tiff::tags::Tag::SamplesPerPixel)
                .unwrap() as u16;
            assert_eq!(spp, want_bands, "{mode:?} band count");
        }
    }
}
