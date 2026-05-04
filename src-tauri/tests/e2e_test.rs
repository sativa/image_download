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
use tiff::tags::Tag;

fn fake_red_tile() -> Bytes {
    let buf: image::RgbaImage =
        image::ImageBuffer::from_pixel(256, 256, image::Rgba([200, 30, 30, 255]));
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(&buf, 256, 256, image::ColorType::Rgba8.into())
        .unwrap();
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
    let tiles: Vec<DownloadedTile> = range
        .iter()
        .map(|c: TileCoord| DownloadedTile {
            coord: c,
            bytes: Some(fake_red_tile()),
        })
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

    // 6. Read back via tiff crate, verify dimensions + tag presence
    let f = std::fs::File::open(&out).unwrap();
    let mut dec = tiff::decoder::Decoder::new(f).unwrap();
    let dims = dec.dimensions().unwrap();
    assert_eq!(dims, (expected_w, expected_h));
    let tp = dec.get_tag_f64_vec(Tag::Unknown(33922)).unwrap();
    // Tiepoint y (world) ≈ bbox_3857[3] (north)
    assert!((tp[4] - bbox_3857[3]).abs() < 1.0);
}
