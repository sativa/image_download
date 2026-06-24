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

    /// The other provider — used as a QC fallback when one source serves a
    /// blank/black tile for a cell.
    pub fn alternate(self) -> SourceKind {
        match self {
            SourceKind::Esri => SourceKind::Google,
            SourceKind::Google => SourceKind::Esri,
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

use std::time::{Duration, Instant};

/// Issue a GET to the URL, return wall-clock time. Errors on non-2xx or network failure.
pub async fn probe_url(url: &str) -> Result<Duration, reqwest::Error> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let start = Instant::now();
    let resp = client.get(url).send().await?;
    resp.error_for_status()?;
    Ok(start.elapsed())
}

/// Result of probing both sources. `_ms` fields are None on probe failure.
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub esri_ms: Option<u64>,
    pub google_ms: Option<u64>,
    pub recommended: SourceKind,
}

/// Probe both Esri and Google in parallel and return latencies plus the
/// recommended (faster, or only-reachable) source. Sample tile is small
/// (continent-scale, z=2) so each probe finishes in ~100 ms.
pub async fn probe_both() -> ProbeReport {
    let sample = TileCoord { x: 0, y: 0, z: 2 };
    let esri_url = url_for_tile(SourceKind::Esri, sample);
    let google_url = url_for_tile(SourceKind::Google, sample);
    let (esri, google) = tokio::join!(probe_url(&esri_url), probe_url(&google_url));

    let esri_ms = esri.as_ref().ok().map(|d| d.as_millis() as u64);
    let google_ms = google.as_ref().ok().map(|d| d.as_millis() as u64);

    let recommended = match (esri_ms, google_ms) {
        (Some(e), Some(g)) if e <= g => SourceKind::Esri,
        (Some(_), Some(_)) => SourceKind::Google,
        (Some(_), None) => SourceKind::Esri,
        (None, Some(_)) => SourceKind::Google,
        (None, None) => SourceKind::Esri, // fallback when both unreachable
    };

    ProbeReport {
        esri_ms,
        google_ms,
        recommended,
    }
}

/// Convenience wrapper that only returns the recommended source. Existing
/// callers (and tests) that don't care about latency keep working.
pub async fn pick_auto() -> SourceKind {
    probe_both().await.recommended
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

    #[tokio::test]
    #[ignore = "real network"]
    async fn pick_auto_returns_some_source() {
        let pick = pick_auto().await;
        assert!(pick == SourceKind::Esri || pick == SourceKind::Google);
    }
}
