//! Core domain modules implementing the satellite-imagery downloader.
//!
//! All modules are Tauri-agnostic — they take plain inputs and return
//! plain outputs. Plan B's commands/ wraps them.

pub mod cog;
pub mod coverage;
pub mod downloader;
pub mod history;
pub mod job;
pub mod sources;
pub mod stitcher;
pub mod tiles;
pub mod vector;
