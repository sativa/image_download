//! Property-style tests for tiles module: roundtrip + range invariants.

use imagery_downloader_lib::core::tiles::*;

fn next_pseudo(seed: &mut u64) -> f64 {
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
            let span_lon = 360.0 / 2_f64.powi(zoom as i32);
            assert!(
                lon - lon_back >= 0.0 && lon - lon_back <= span_lon + 1e-9,
                "lon={lon} z={zoom} x={x} back={lon_back} span={span_lon}"
            );
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
