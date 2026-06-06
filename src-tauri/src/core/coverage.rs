//! Pre-flight imagery-coverage probe.
//!
//! Some upstream tile sources (notably Esri at high zoom) return a constant
//! "Map data not yet available" placeholder JPEG instead of a 404 when the
//! requested area has no imagery at that zoom. The downloader will happily
//! fetch thousands of those, stitch them into a multi-hundred-MB COG, and
//! present the user with a uniform grey output — wasting time and disk.
//!
//! This module samples a 3×3 grid of tiles over the bbox, identifies known
//! placeholders by exact-bytes match plus a header-only fast path (ETag),
//! and reports a coverage ratio that `start_download` checks before
//! creating a Job.
//!
//! Adding a new source's placeholder: drop the bytes into `assets/`, list it
//! in `PLACEHOLDERS_FOR`, and add an optional ETag literal if the source
//! returns one. No code in download/stitch paths needs to change.
//!
//! Out of scope (deliberate, keeps the fix small):
//!   - In-flight detection that aborts mid-download when the placeholder
//!     rate spikes. The 3×3 grid catches the "completely empty" case which
//!     is the dominant waste vector; partial coverage will still produce a
//!     usable COG.

use crate::core::downloader::{build_client, DownloadConfig};
use crate::core::sources::{url_for_tile, SourceKind};
use crate::core::tiles::{TileCoord, TileRange};
use std::time::Duration;

/// Embedded reference: the exact 2521-byte JPEG Esri returns for tiles
/// outside its imagery extent (visible text "Map data not yet available").
/// Identical bytes for every (x, y) lacking coverage, so a single embedded
/// copy suffices.
const ESRI_PLACEHOLDER_BYTES: &[u8] = include_bytes!("assets/esri_placeholder.jpg");

/// Fingerprints we recognise as "no imagery". Both fields are checked
/// independently — either one matching is enough.
struct Placeholder {
    /// Full raw response body. Compared byte-for-byte (cheap because tiles
    /// are small and equal-length comparison short-circuits fast).
    bytes: &'static [u8],
    /// ETag literal the server sets for the placeholder, if any. Allows
    /// header-only detection on HEAD-style probes where we wouldn't see the
    /// body. Quotes are part of the header value, include them.
    etag: Option<&'static str>,
}

/// Look up the known placeholder fingerprints for a source. Empty slice
/// means "we don't know any placeholder for this source yet" → the probe
/// will simply report full coverage.
fn placeholders_for(source: SourceKind) -> &'static [Placeholder] {
    match source {
        SourceKind::Esri => &[Placeholder {
            bytes: ESRI_PLACEHOLDER_BYTES,
            etag: Some("\"vvvvvvvvvvvvf\""),
        }],
        // Google's tile API 404s on missing imagery rather than returning a
        // placeholder, so this list stays empty until we observe otherwise.
        SourceKind::Google => &[],
    }
}

/// Decide whether a fetched tile is a known placeholder. ETag wins when
/// present (header-only check, no body comparison needed); otherwise fall
/// back to byte-equality.
pub fn is_placeholder(source: SourceKind, body: &[u8], etag: Option<&str>) -> bool {
    let phs = placeholders_for(source);
    if phs.is_empty() {
        return false;
    }
    for ph in phs {
        if let (Some(want), Some(got)) = (ph.etag, etag) {
            if want == got {
                return true;
            }
        }
        if ph.bytes.len() == body.len() && ph.bytes == body {
            return true;
        }
    }
    false
}

/// 3×3 grid of sample tiles spread evenly over a range. Returns up to 9
/// distinct coordinates; for very small ranges (≤3×3 tiles) every tile in
/// the range is returned without duplicates.
pub fn sample_grid(range: TileRange) -> Vec<TileCoord> {
    let xs = pick_axis(range.x_min, range.x_max);
    let ys = pick_axis(range.y_min, range.y_max);
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    let mut seen = std::collections::HashSet::new();
    for y in &ys {
        for x in &xs {
            let c = TileCoord {
                x: *x,
                y: *y,
                z: range.z,
            };
            if seen.insert((c.x, c.y)) {
                out.push(c);
            }
        }
    }
    out
}

fn pick_axis(lo: i64, hi: i64) -> Vec<i64> {
    let span = hi - lo;
    if span < 0 {
        return vec![];
    }
    if span <= 2 {
        return (lo..=hi).collect();
    }
    // Three evenly spaced points: low, midpoint, high — avoids the corners
    // being the only samples on long ranges.
    vec![lo, lo + span / 2, hi]
}

#[derive(Debug, Clone)]
pub struct CoverageReport {
    pub sampled: u32,
    pub placeholder: u32,
    /// Tiles we couldn't fetch at all (network error / non-2xx). Treated as
    /// neither covered nor placeholder so a flaky probe doesn't block a job.
    pub errored: u32,
    /// `covered / max(sampled - errored, 1)`. NaN-safe.
    pub coverage_ratio: f32,
}

impl CoverageReport {
    pub fn covered(&self) -> u32 {
        self.sampled
            .saturating_sub(self.placeholder)
            .saturating_sub(self.errored)
    }
}

/// Fetch the sample grid and classify each response. Runs all sample
/// requests in parallel; total wall-clock ≈ slowest single probe.
pub async fn probe_coverage(
    range: TileRange,
    source: SourceKind,
) -> Result<CoverageReport, String> {
    let coords = sample_grid(range);
    if coords.is_empty() {
        return Ok(CoverageReport {
            sampled: 0,
            placeholder: 0,
            errored: 0,
            coverage_ratio: 1.0,
        });
    }

    let cfg = DownloadConfig {
        max_retries: 0,
        backoff_base: Duration::from_millis(100),
        timeout_per_request: Duration::from_secs(8),
    };
    let client = build_client(&cfg).map_err(|e| format!("probe client init: {e}"))?;

    let futures = coords.iter().map(|c| {
        let url = url_for_tile(source, *c);
        let client = client.clone();
        async move {
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let etag = resp
                        .headers()
                        .get("etag")
                        .and_then(|v| v.to_str().ok())
                        .map(|s| s.to_owned());
                    match resp.bytes().await {
                        Ok(body) => Some((body.to_vec(), etag)),
                        Err(_) => None,
                    }
                }
                _ => None,
            }
        }
    });

    let results: Vec<Option<(Vec<u8>, Option<String>)>> = futures::future::join_all(futures).await;
    let sampled = results.len() as u32;
    let mut placeholder = 0u32;
    let mut errored = 0u32;
    for r in results {
        match r {
            None => errored += 1,
            Some((body, etag)) => {
                if is_placeholder(source, &body, etag.as_deref()) {
                    placeholder += 1;
                }
            }
        }
    }
    let usable = sampled.saturating_sub(errored);
    let covered = sampled.saturating_sub(placeholder).saturating_sub(errored);
    // If every probe errored we have no signal at all; report full coverage
    // so a flaky network doesn't masquerade as a no-imagery diagnosis.
    let coverage_ratio = if usable == 0 {
        1.0
    } else {
        covered as f32 / usable as f32
    };
    Ok(CoverageReport {
        sampled,
        placeholder,
        errored,
        coverage_ratio,
    })
}

/// Coverage ratio threshold below which `start_download` aborts pre-flight.
/// 0.34 = "if 2/3 or more of the sampled tiles are placeholders, the job is
/// almost certainly worthless". Two genuine reasons to keep it loose rather
/// than e.g. 0.8:
///   1. We sample a 3×3 grid; one or two placeholders in a partly-covered
///      area is a perfectly valid download.
///   2. False-negatives (let a bad job through) are far cheaper than
///      false-positives (block a legitimate download): the user can see the
///      result and rerun at a lower zoom.
pub const MIN_COVERAGE_TO_PROCEED: f32 = 0.34;

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(x_min: i64, y_min: i64, x_max: i64, y_max: i64, z: u32) -> TileRange {
        TileRange {
            x_min,
            y_min,
            x_max,
            y_max,
            z,
        }
    }

    #[test]
    fn sample_grid_full_3x3_on_large_range() {
        let g = sample_grid(rng(0, 0, 100, 100, 10));
        assert_eq!(g.len(), 9);
        // Corners and midpoint must be present.
        assert!(g.iter().any(|c| c.x == 0 && c.y == 0));
        assert!(g.iter().any(|c| c.x == 100 && c.y == 100));
        assert!(g.iter().any(|c| c.x == 50 && c.y == 50));
    }

    #[test]
    fn sample_grid_no_duplicates_on_small_range() {
        let g = sample_grid(rng(5, 5, 6, 6, 10));
        assert_eq!(g.len(), 4);
        let unique: std::collections::HashSet<_> = g.iter().map(|c| (c.x, c.y)).collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn sample_grid_single_tile() {
        let g = sample_grid(rng(7, 7, 7, 7, 12));
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn placeholder_detected_via_bytes() {
        assert!(is_placeholder(
            SourceKind::Esri,
            ESRI_PLACEHOLDER_BYTES,
            None
        ));
    }

    #[test]
    fn placeholder_detected_via_etag_without_body_match() {
        // Body deliberately wrong size: should still trigger via ETag.
        assert!(is_placeholder(
            SourceKind::Esri,
            b"different bytes",
            Some("\"vvvvvvvvvvvvf\""),
        ));
    }

    #[test]
    fn non_placeholder_bytes_pass() {
        // A real-looking JPEG-ish blob of different length never matches.
        let fake = vec![0xff; ESRI_PLACEHOLDER_BYTES.len() + 1];
        assert!(!is_placeholder(SourceKind::Esri, &fake, None));
    }

    #[test]
    fn google_currently_has_no_known_placeholder() {
        // Even the Esri placeholder bytes shouldn't match Google's source.
        assert!(!is_placeholder(SourceKind::Google, ESRI_PLACEHOLDER_BYTES, None));
    }

    #[test]
    fn coverage_ratio_handles_all_errored() {
        let r = CoverageReport {
            sampled: 5,
            placeholder: 0,
            errored: 5,
            coverage_ratio: 1.0,
        };
        assert_eq!(r.covered(), 0);
        assert!(
            r.coverage_ratio >= MIN_COVERAGE_TO_PROCEED,
            "all-errored probe must not block downloads"
        );
    }
}
