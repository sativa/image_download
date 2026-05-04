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
    pub output_path: String,
    pub ok: bool,
    pub duration_sec: f64,
    pub total_tiles: u32,
    pub failed_tiles: u32,
    pub output_size_mb: f64,
    pub finished_at: String,
}

pub struct Store {
    path: PathBuf,
    inner: Mutex<Vec<HistoryEntry>>,
}

const MAX: usize = 10;

impl Store {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            let bytes = fs::read(&path)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Self {
            path,
            inner: Mutex::new(inner),
        })
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
