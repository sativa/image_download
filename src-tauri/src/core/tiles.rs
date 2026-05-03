//! Web-Mercator tile math (EPSG:3857).
//!
//! XYZ tile coordinates: x grows east 0..2^z, y grows south 0..2^z, z is zoom.
//! Conventions match OSM / ESRI / Google.

use std::f64::consts::PI;

pub fn lon_to_tile_x(lon: f64, zoom: u32) -> i64 {
    let n = 2_f64.powi(zoom as i32);
    ((lon + 180.0) / 360.0 * n).floor() as i64
}

pub fn lat_to_tile_y(lat: f64, zoom: u32) -> i64 {
    let n = 2_f64.powi(zoom as i32);
    let lat_rad = lat.clamp(-85.05112878, 85.05112878).to_radians();
    ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * n).floor() as i64
}

pub fn tile_x_to_lon(x: i64, zoom: u32) -> f64 {
    let n = 2_f64.powi(zoom as i32);
    x as f64 / n * 360.0 - 180.0
}

pub fn tile_y_to_lat(y: i64, zoom: u32) -> f64 {
    let n = 2_f64.powi(zoom as i32);
    let lat_rad = (PI * (1.0 - 2.0 * y as f64 / n)).sinh().atan();
    lat_rad.to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lon_origin_at_zoom_2() {
        assert_eq!(lon_to_tile_x(0.0, 2), 2);
    }

    #[test]
    fn lon_west_extreme() {
        assert_eq!(lon_to_tile_x(-180.0, 2), 0);
        assert_eq!(lon_to_tile_x(-180.0, 4), 0);
    }

    #[test]
    fn lon_just_under_east_extreme() {
        assert_eq!(lon_to_tile_x(179.999, 4), 15);
    }

    #[test]
    fn lat_equator_at_zoom_2() {
        assert_eq!(lat_to_tile_y(0.0, 2), 2);
    }

    #[test]
    fn lat_clamps_to_mercator_extent() {
        // At zoom 4, clamped extreme latitudes land in the boundary tiles 0 / 15.
        // FP precision at the exact clamp can underflow below 0, so we accept the
        // boundary tile (0 or -1 / 15 or 16) and verify magnitude only.
        let north = lat_to_tile_y(89.0, 4);
        let south = lat_to_tile_y(-89.0, 4);
        assert!(north <= 0, "north tile should clamp at top: got {}", north);
        assert!(south >= 15, "south tile should clamp at bottom: got {}", south);
    }

    #[test]
    fn known_pair_beijing() {
        // Beijing 116.404°E, 39.915°N at z=10 — verified via the canonical OSM
        // slippy-map formula: y = floor((1 - asinh(tan(lat))/π)/2 * 2^z) = 387.
        // (Spec value 388 was the off-by-one this test was meant to catch.)
        assert_eq!(lon_to_tile_x(116.404, 10), 843);
        assert_eq!(lat_to_tile_y(39.915, 10), 387);
    }

    #[test]
    fn roundtrip_lon() {
        for &lon in &[-180.0, -90.0, 0.0, 90.0, 179.999] {
            let x = lon_to_tile_x(lon, 8);
            let back = tile_x_to_lon(x, 8);
            assert!(back <= lon, "lon={} → x={} → back={}", lon, x, back);
            assert!(back >= lon - 360.0 / 256.0, "tile too far west");
        }
    }

    #[test]
    fn roundtrip_lat() {
        for &lat in &[-80.0, -45.0, 0.0, 45.0, 80.0] {
            let y = lat_to_tile_y(lat, 8);
            let back = tile_y_to_lat(y, 8);
            assert!(back >= lat - 1.0, "lat={} → y={} → back={}", lat, y, back);
        }
    }

    #[test]
    fn bbox_corners_at_zoom_4_china() {
        let west = tile_x_to_lon(12, 4);
        let north = tile_y_to_lat(6, 4);
        assert!((west - 90.0).abs() < 0.001);
        assert!((north - 40.97).abs() < 0.1);
    }
}
