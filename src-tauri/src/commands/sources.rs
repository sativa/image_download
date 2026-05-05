//! Source-selection IPC commands.
//!
//! `probe_sources` lets the UI display real latency numbers and lets the
//! user make an informed choice; the underlying `pick_auto` is also used
//! server-side when `start_download` receives `source: "auto"`.

use crate::core::sources::{probe_both, ProbeReport};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ProbeReportDto {
    /// Esri latency in milliseconds; null when the probe failed.
    pub esri_ms: Option<u64>,
    /// Google latency in milliseconds; null when the probe failed.
    pub google_ms: Option<u64>,
    /// "esri" or "google". Falls back to "esri" if both probes fail.
    pub recommended: String,
}

impl From<ProbeReport> for ProbeReportDto {
    fn from(r: ProbeReport) -> Self {
        Self {
            esri_ms: r.esri_ms,
            google_ms: r.google_ms,
            recommended: r.recommended.as_str().to_string(),
        }
    }
}

/// Probe both Esri and Google with a tiny sample tile and report latencies.
/// Both probes share a 5-second timeout; the slower / unreachable one
/// returns `None` for its `*_ms` field while the other still resolves.
#[tauri::command]
pub async fn probe_sources() -> Result<ProbeReportDto, String> {
    Ok(probe_both().await.into())
}
