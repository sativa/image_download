//! Core domain modules implementing the satellite-imagery downloader.
//!
//! All modules are Tauri-agnostic — they take plain inputs and return
//! plain outputs. Plan B's commands/ wraps them.

pub mod tiles;
pub mod sources;
pub mod downloader;
pub mod stitcher;
pub mod cog;
pub mod vector;
pub mod history;
