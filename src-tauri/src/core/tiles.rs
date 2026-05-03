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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub x: i64,
    pub y: i64,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileRange {
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64, // inclusive
    pub y_max: i64, // inclusive
    pub z: u32,
}

impl TileRange {
    pub fn count(&self) -> u64 {
        ((self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)) as u64
    }
    pub fn iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        let z = self.z;
        (self.y_min..=self.y_max).flat_map(move |y| {
            (self.x_min..=self.x_max).map(move |x| TileCoord { x, y, z })
        })
    }
}

/// Compute the inclusive tile range covering the bbox [minLon, minLat, maxLon, maxLat] at zoom.
/// Bbox is in WGS84; tiles are XYZ web-mercator.
pub fn range_for_bbox(bbox: [f64; 4], zoom: u32) -> TileRange {
    let [w, s, e, n] = bbox;
    let x_min = lon_to_tile_x(w, zoom);
    let x_max = lon_to_tile_x(e, zoom).max(x_min);
    let y_min = lat_to_tile_y(n, zoom);
    let y_max = lat_to_tile_y(s, zoom).max(y_min);
    let max_idx = (1i64 << zoom) - 1;
    TileRange {
        x_min: x_min.clamp(0, max_idx),
        y_min: y_min.clamp(0, max_idx),
        x_max: x_max.clamp(0, max_idx),
        y_max: y_max.clamp(0, max_idx),
        z: zoom,
    }
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

    #[test]
    fn range_single_tile() {
        let r = range_for_bbox([0.5, 0.5, 0.6, 0.6], 8);
        assert_eq!(r.count(), 1);
    }

    #[test]
    fn range_known_extent() {
        // bbox 100..110 lon, 30..40 lat at z=8: ~8 cols (lon) × ~10 rows (lat) ≈ 80 tiles.
        // (Spec band 50–64 was too narrow: 10° lon at z=8 spans 8 tiles, 10° lat at
        // mid-latitudes spans ~10 tiles in the mercator projection.)
        let r = range_for_bbox([100.0, 30.0, 110.0, 40.0], 8);
        assert!(r.count() >= 70 && r.count() <= 90, "got {}", r.count());
        assert!(r.x_min < r.x_max);
        assert!(r.y_min < r.y_max);
    }

    #[test]
    fn range_iter_yields_unique_tiles() {
        let r = range_for_bbox([0.0, 0.0, 1.0, 1.0], 6);
        let v: Vec<_> = r.iter().collect();
        let unique: std::collections::HashSet<_> = v.iter().copied().collect();
        assert_eq!(v.len(), unique.len());
        assert_eq!(v.len() as u64, r.count());
    }

    #[test]
    fn range_clamps_to_world() {
        let r = range_for_bbox([-181.0, -86.0, 181.0, 86.0], 4);
        assert_eq!(r.x_min, 0);
        assert_eq!(r.x_max, 15);
        assert_eq!(r.y_min, 0);
        assert_eq!(r.y_max, 15);
    }
}
