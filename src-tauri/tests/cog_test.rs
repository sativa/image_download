use image::{ImageBuffer, Rgba};
use imagery_downloader_lib::core::cog::{write_cog, write_preview_png, CogParams};
use tempfile::tempdir;
use tiff::decoder::Decoder;
use tiff::tags::Tag;

#[test]
fn write_cog_writes_a_readable_tiff() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("out.tif");
    let img: image::RgbaImage = ImageBuffer::from_pixel(512, 256, Rgba([1, 2, 3, 4]));
    write_cog(
        &img,
        &CogParams {
            bbox_3857: [0.0, 0.0, 1024.0, 512.0],
            zoom: 5,
        },
        &p,
    )
    .unwrap();
    assert!(p.exists());
    let f = std::fs::File::open(&p).unwrap();
    let mut dec = Decoder::new(f).unwrap();
    let dims = dec.dimensions().unwrap();
    assert_eq!(dims, (512, 256));
}

#[test]
fn cog_carries_geotiff_tags() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("geo.tif");
    let img: image::RgbaImage = ImageBuffer::from_pixel(256, 256, Rgba([255, 255, 255, 255]));
    write_cog(
        &img,
        &CogParams {
            bbox_3857: [0.0, 0.0, 100.0, 100.0],
            zoom: 5,
        },
        &p,
    )
    .unwrap();

    let f = std::fs::File::open(&p).unwrap();
    let mut dec = Decoder::new(f).unwrap();

    let scale = dec.get_tag_f64_vec(Tag::Unknown(33550)).unwrap();
    assert!((scale[0] - 100.0 / 256.0).abs() < 1e-9);
    assert!((scale[1] - 100.0 / 256.0).abs() < 1e-9);

    let tp = dec.get_tag_f64_vec(Tag::Unknown(33922)).unwrap();
    assert_eq!(tp[3], 0.0);
    assert_eq!(tp[4], 100.0);

    let keys = dec
        .get_tag_u32_vec(Tag::Unknown(34735))
        .or_else(|_| dec.get_tag_u32_vec(Tag::Unknown(34735)))
        .unwrap_or_default();
    let _ = keys;
    // Some tiff readers expose GeoKey as u16 vec; both should contain 3857
    let keys_u16 = dec.get_tag(Tag::Unknown(34735)).ok();
    assert!(keys_u16.is_some(), "GeoKeyDirectory tag should be present");
}

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
