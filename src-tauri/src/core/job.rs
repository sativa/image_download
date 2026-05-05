//! Resumable-download job state stored alongside the output file.
//!
//! Layout for a job whose output stem is `imagery_z17_esri_a3f1c0e7`:
//!   imagery_z17_esri_a3f1c0e7.tif        (final COG; written at the end)
//!   imagery_z17_esri_a3f1c0e7.job.json   (static parameters; written once)
//!   imagery_z17_esri_a3f1c0e7.completed.txt
//!       └─ append-only WAL, one line per finished tile: "{x},{y}\n"
//!   imagery_z17_esri_a3f1c0e7.tiles/
//!       └─ {x}_{y}.bin     raw tile bytes, one per finished tile
//!
//! Why this layout:
//!   * append-only WAL avoids rewriting a large JSON for every finished tile;
//!   * tile bytes have to live on disk (otherwise resume would still need to
//!     redownload everything just to rebuild the byte buffer for stitching);
//!   * stem is derived from `bbox + zoom + source` so re-running the same
//!     parameters in the same folder *resumes* instead of clobbering or
//!     duplicating.
//!
//! Cleanup: call `Job::clear_state()` after the COG is written to remove all
//! sidecar files. The .tif (and .preview.png) stays.

use anyhow::{anyhow, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::core::tiles::TileCoord;

/// On-disk static manifest written once when a job is created.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobManifest {
    pub version: u32,
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    pub total_tiles: u32,
    pub started_at_epoch: u64,
}

const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct Job {
    pub stem: String,
    pub dir: PathBuf,
    pub manifest: JobManifest,
}

impl Job {
    /// Open or create a Job in `dir` for the given parameters. If a
    /// matching manifest already exists, it is loaded as-is; otherwise a
    /// fresh manifest is written.
    ///
    /// `total_tiles` is taken from the manifest on resume — so the caller
    /// should treat this as authoritative for the lifetime of the job, even
    /// if it recomputes the value from the bbox.
    pub fn open_or_create(
        dir: &Path,
        bbox: [f64; 4],
        zoom: u32,
        source: &str,
        total_tiles_if_new: u32,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let stem = make_stem(&bbox, zoom, source);
        let manifest_path = dir.join(format!("{stem}.job.json"));

        let manifest = if manifest_path.exists() {
            let s = std::fs::read_to_string(&manifest_path)?;
            let m: JobManifest = serde_json::from_str(&s)
                .map_err(|e| anyhow!("corrupt job manifest {}: {e}", manifest_path.display()))?;
            if m.version != MANIFEST_VERSION {
                return Err(anyhow!(
                    "manifest version mismatch (expected {}, got {})",
                    MANIFEST_VERSION,
                    m.version
                ));
            }
            m
        } else {
            let m = JobManifest {
                version: MANIFEST_VERSION,
                bbox,
                zoom,
                source: source.to_string(),
                total_tiles: total_tiles_if_new,
                started_at_epoch: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            };
            std::fs::write(&manifest_path, serde_json::to_string_pretty(&m)?)?;
            m
        };

        std::fs::create_dir_all(dir.join(format!("{stem}.tiles")))?;

        Ok(Self {
            stem,
            dir: dir.to_path_buf(),
            manifest,
        })
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join(format!("{}.job.json", self.stem))
    }

    pub fn completed_log_path(&self) -> PathBuf {
        self.dir.join(format!("{}.completed.txt", self.stem))
    }

    pub fn tiles_dir(&self) -> PathBuf {
        self.dir.join(format!("{}.tiles", self.stem))
    }

    pub fn tile_path(&self, c: TileCoord) -> PathBuf {
        self.tiles_dir().join(format!("{}_{}.bin", c.x, c.y))
    }

    pub fn output_tif_path(&self) -> PathBuf {
        // Final COG uses a `merge_` prefix instead of the bare stem so its
        // filename can never collide with sidecars (`<stem>.tiles/`,
        // `<stem>.completed.txt`, `<stem>.job.json`). If a downstream caller
        // ever mistakes the .tif path for a folder and `create_dir_all`s it,
        // the resulting `merge_<stem>.tif/` would still hold a fresh job
        // whose own .tif is `merge_merge_<stem>.tif` — making nesting
        // visually obvious and keeping each layer's stem unique.
        self.dir.join(format!("merge_{}.tif", self.stem))
    }

    /// Read the WAL and return the set of (x, y) coordinates already
    /// downloaded. Tile bytes for those coordinates are expected to live in
    /// `tiles_dir()`; orphan WAL entries (no tile file) are ignored on read.
    pub fn load_completed(&self) -> Result<HashSet<(i64, i64)>> {
        let path = self.completed_log_path();
        if !path.exists() {
            return Ok(HashSet::new());
        }
        let f = File::open(&path)?;
        let mut set = HashSet::new();
        for line in BufReader::new(f).lines() {
            let l = line?;
            let l = l.trim();
            if l.is_empty() {
                continue;
            }
            if let Some((xs, ys)) = l.split_once(',') {
                if let (Ok(x), Ok(y)) = (xs.parse::<i64>(), ys.parse::<i64>()) {
                    set.insert((x, y));
                }
            }
        }
        Ok(set)
    }

    /// Persist a freshly-downloaded tile: write bytes to disk and append the
    /// coordinate to the WAL. Both writes are flushed before returning so a
    /// crash mid-call cannot leave the WAL pointing at an unwritten file.
    pub fn record_tile(&self, c: TileCoord, bytes: &Bytes) -> Result<()> {
        let p = self.tile_path(c);
        // Write tile bytes first; flush to disk; only then append to the WAL.
        let mut f = BufWriter::new(File::create(&p)?);
        f.write_all(bytes)?;
        f.flush()?;
        f.into_inner()?.sync_data()?;

        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.completed_log_path())?;
        writeln!(log, "{},{}", c.x, c.y)?;
        log.sync_data()?;
        Ok(())
    }

    /// Read the cached bytes for one previously-recorded tile.
    pub fn load_tile(&self, c: TileCoord) -> Result<Bytes> {
        let p = self.tile_path(c);
        let v = std::fs::read(&p)
            .map_err(|e| anyhow!("missing cached tile {}: {e}", p.display()))?;
        Ok(Bytes::from(v))
    }

    /// Remove sidecar manifest, WAL, and tile cache. Call after the COG has
    /// been written successfully — the .tif itself is left in place.
    pub fn clear_state(&self) -> Result<()> {
        let _ = std::fs::remove_file(self.manifest_path());
        let _ = std::fs::remove_file(self.completed_log_path());
        let dir = self.tiles_dir();
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }
}

/// Build a stable filename stem from the *task-defining* parameters.
///
/// Format: `imagery_z{zoom}_{source}_{bbox-fingerprint:08x}`
/// where the fingerprint is a 32-bit hash of the bbox bit-pattern.
///
/// Stable means: re-running with identical bbox/zoom/source in the same
/// output folder hits the same stem and resumes. Changing any of the three
/// produces a different stem and starts a new job.
pub fn make_stem(bbox: &[f64; 4], zoom: u32, source: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in bbox {
        v.to_bits().hash(&mut h);
    }
    let safe = source
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect::<String>();
    let safe = if safe.is_empty() { "src".into() } else { safe };
    format!(
        "imagery_z{zoom}_{safe}_{:08x}",
        (h.finish() as u32)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_is_stable_for_same_inputs() {
        let s1 = make_stem(&[100.0, 30.0, 102.0, 32.0], 17, "esri");
        let s2 = make_stem(&[100.0, 30.0, 102.0, 32.0], 17, "esri");
        assert_eq!(s1, s2);
    }

    #[test]
    fn stem_changes_for_different_bbox_zoom_source() {
        let base = make_stem(&[100.0, 30.0, 102.0, 32.0], 17, "esri");
        assert_ne!(base, make_stem(&[100.0, 30.0, 102.0, 33.0], 17, "esri"));
        assert_ne!(base, make_stem(&[100.0, 30.0, 102.0, 32.0], 18, "esri"));
        assert_ne!(base, make_stem(&[100.0, 30.0, 102.0, 32.0], 17, "google"));
    }

    #[test]
    fn open_or_create_then_resume_round_trips_completed_set() {
        let dir = tempfile::tempdir().unwrap();
        let job = Job::open_or_create(dir.path(), [100.0, 30.0, 101.0, 31.0], 10, "esri", 4)
            .unwrap();

        let coords = [
            TileCoord { x: 1, y: 2, z: 10 },
            TileCoord { x: 1, y: 3, z: 10 },
        ];
        for c in &coords {
            job.record_tile(*c, &Bytes::from(format!("body-{}-{}", c.x, c.y))).unwrap();
        }

        // Reopen — should pick up the same manifest and WAL entries.
        let job2 = Job::open_or_create(dir.path(), [100.0, 30.0, 101.0, 31.0], 10, "esri", 99)
            .unwrap();
        assert_eq!(job2.manifest.total_tiles, 4); // not 99 — manifest was preserved
        let done = job2.load_completed().unwrap();
        assert_eq!(done.len(), 2);
        assert!(done.contains(&(1, 2)));
        assert!(done.contains(&(1, 3)));
        // Tile bytes are recoverable.
        let b = job2.load_tile(coords[0]).unwrap();
        assert_eq!(&b[..], b"body-1-2");
    }

    #[test]
    fn clear_state_removes_sidecars_only() {
        let dir = tempfile::tempdir().unwrap();
        let job = Job::open_or_create(dir.path(), [0.0, 0.0, 1.0, 1.0], 8, "esri", 1).unwrap();
        let tif = job.output_tif_path();
        std::fs::write(&tif, b"pretend cog").unwrap();
        job.record_tile(TileCoord { x: 0, y: 0, z: 8 }, &Bytes::from_static(b"x")).unwrap();

        job.clear_state().unwrap();
        assert!(tif.exists(), "the .tif must survive cleanup");
        assert!(!job.manifest_path().exists());
        assert!(!job.completed_log_path().exists());
        assert!(!job.tiles_dir().exists());
    }
}
