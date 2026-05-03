//! XYZ tile source URL templates and auto-selection.

use crate::core::tiles::TileCoord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    Esri,
    Google,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Esri => "esri",
            SourceKind::Google => "google",
        }
    }

    pub fn parse(s: &str) -> Option<SourceKind> {
        match s {
            "esri" => Some(SourceKind::Esri),
            "google" => Some(SourceKind::Google),
            _ => None,
        }
    }
}

/// Build the XYZ URL for one tile from one source.
pub fn url_for_tile(s: SourceKind, t: TileCoord) -> String {
    match s {
        SourceKind::Esri => format!(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            z = t.z, x = t.x, y = t.y,
        ),
        SourceKind::Google => format!(
            "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            x = t.x, y = t.y, z = t.z,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tiles::TileCoord;

    #[test]
    fn esri_url_has_z_y_x_order() {
        let u = url_for_tile(SourceKind::Esri, TileCoord { x: 1, y: 2, z: 3 });
        assert!(u.ends_with("/3/2/1"));
    }

    #[test]
    fn google_url_has_query_params() {
        let u = url_for_tile(SourceKind::Google, TileCoord { x: 1, y: 2, z: 3 });
        assert!(u.contains("x=1") && u.contains("y=2") && u.contains("z=3"));
    }

    #[test]
    fn parse_roundtrip() {
        for s in [SourceKind::Esri, SourceKind::Google] {
            assert_eq!(SourceKind::parse(s.as_str()), Some(s));
        }
        assert_eq!(SourceKind::parse("bing"), None);
    }
}
