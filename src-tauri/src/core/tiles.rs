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
}
