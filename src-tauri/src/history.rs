//! Persistent history of recent download parameters.
//!
//! This module is logically part of Plan A's `core::history`, but is
//! implemented during Plan C so the mock UI can demonstrate persistence.
//! Plan A will move it under `core::history` without API changes.
//!
//! NOTE: This is a stub with public types only. Task 6.1 fills in the
//! real `Store` impl with file persistence + tests.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// Stub. Real impl in Task 6.1.
pub fn history_path() -> PathBuf {
    PathBuf::from("/tmp/imagery-downloader-history.json")
}
