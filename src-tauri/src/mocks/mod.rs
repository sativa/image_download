//! Mock backend implementations.
//!
//! Plan B will replace this entire directory with real `core::*` calls.
//! Frontend invokes (in src/lib/ipc.ts) MUST NOT change between mock
//! and real — only the Rust handlers swap.

pub mod commands;
pub mod runner;
