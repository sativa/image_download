use imagery_downloader_lib::core::cog::{write_cog, CogParams};
use image::{ImageBuffer, Rgba};
use tempfile::tempdir;
use tiff::decoder::Decoder;

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
