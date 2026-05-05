//! Persistent history of recent download parameters.
//!
//! Logically part of Plan A's `core::history`; implemented during Plan C
//! so the mock UI can demonstrate persistence. Plan A will relocate this
//! module without API changes.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    /// User-chosen output **folder** for this run. Resume re-uses this verbatim
    /// as `input.outputPath`, which is why it must be a directory — never a
    /// .tif file path. Old history.json files (where this used to be the
    /// final .tif full-path) are migrated on `Store::open`.
    pub output_path: String,
    /// Legacy success flag — true iff the COG was written successfully.
    /// Kept for forward/backward compat with older history.json files;
    /// new code should branch on `status`.
    pub ok: bool,
    pub duration_sec: f64,
    pub total_tiles: u32,
    pub failed_tiles: u32,
    pub output_size_mb: f64,
    pub finished_at: String,
    /// Tiles already on disk when this entry was last persisted. Lets the UI
    /// show a meaningful "60% done" for in-progress / cancelled entries.
    /// Defaults to 0 for entries written before this field existed.
    #[serde(default)]
    pub completed_tiles: u32,
    /// Lifecycle state. Older entries don't have this field; `default_status`
    /// falls back to "completed".
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_status() -> String {
    // Only used when deserialising an old history.json that pre-dates this
    // field. New entries always set `status` explicitly.
    "completed".to_string()
}

impl HistoryEntry {
    pub const STATUS_IN_PROGRESS: &'static str = "in_progress";
    pub const STATUS_COMPLETED: &'static str = "completed";
    pub const STATUS_CANCELLED: &'static str = "cancelled";
    pub const STATUS_FAILED: &'static str = "failed";
}

pub struct Store {
    path: PathBuf,
    inner: Mutex<Vec<HistoryEntry>>,
}

const MAX: usize = 10;

impl Store {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut inner: Vec<HistoryEntry> = if path.exists() {
            let bytes = fs::read(&path)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };

        // Migration: older builds wrote `output_path` as the final
        // `.tif` file. The "nested-folder" bug also produced paths like
        // `…/yuzhong/imagery.tif/imagery.tif` — multiple layers of
        // `.tif`-named directories. Loop-strip until what remains looks
        // like a directory (or we hit the filesystem root).
        let mut migrated = false;
        for e in inner.iter_mut() {
            while looks_like_file_path(&e.output_path) {
                let parent = parent_dir(&e.output_path);
                if parent == e.output_path {
                    break; // can't go higher
                }
                e.output_path = parent;
                migrated = true;
            }
        }

        let store = Self {
            path,
            inner: Mutex::new(inner),
        };
        if migrated {
            // best-effort: don't fail Store::open if rewrite hits a transient IO error
            let snap = store.inner.lock().unwrap().clone();
            let _ = store.persist(&snap);
        }
        Ok(store)
    }

    pub fn list(&self) -> Vec<HistoryEntry> {
        self.inner.lock().unwrap().clone()
    }

    pub fn add(&self, entry: HistoryEntry) -> io::Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.retain(|e| !(e.bbox == entry.bbox && e.zoom == entry.zoom && e.source == entry.source));
        g.insert(0, entry);
        if g.len() > MAX {
            g.truncate(MAX);
        }
        self.persist(&g)?;
        Ok(())
    }

    pub fn clear(&self) -> io::Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.clear();
        self.persist(&g)?;
        Ok(())
    }

    fn persist(&self, list: &[HistoryEntry]) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(list).unwrap())?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

pub fn default_path() -> PathBuf {
    use directories::ProjectDirs;
    if let Some(d) = ProjectDirs::from("com", "zhangfeng", "imagery-downloader") {
        d.data_dir().join("history.json")
    } else {
        std::env::temp_dir().join("imagery-downloader-history.json")
    }
}

/// True if the path appears to be a file (extension `.tif` / `.tiff`),
/// not a directory. Used by the on-load migration to rewrite legacy entries.
fn looks_like_file_path(s: &str) -> bool {
    let p = Path::new(s);
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()),
        Some(ref e) if e == "tif" || e == "tiff"
    )
}

/// Strip the last path component. `/a/b/c.tif` → `/a/b`. Falls back to the
/// original string if there's no separator (no parent to strip).
fn parent_dir(s: &str) -> String {
    Path::new(s)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| s.to_string())
}
