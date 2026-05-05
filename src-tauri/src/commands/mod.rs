//! Tauri command handlers — wire `core::*` modules into the IPC surface.
//!
//! IPC contract is identical to the previous `mocks/` directory; only the
//! implementation behind each handler is real. See `core/README.md`.

pub mod download;
pub mod history;
pub mod runner;
pub mod sources;
pub mod vector;
