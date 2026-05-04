//! History persistence Tauri commands. Wraps `core::history::Store`.

use crate::core::history::{default_path, HistoryEntry, Store};
use std::sync::OnceLock;

static STORE: OnceLock<Store> = OnceLock::new();

fn store() -> &'static Store {
    STORE.get_or_init(|| Store::open(default_path()).expect("open history store"))
}

#[tauri::command]
pub fn list_history() -> Vec<HistoryEntry> {
    store().list()
}

#[tauri::command]
pub fn clear_history() -> Result<serde_json::Value, String> {
    store().clear().map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "ok": true }))
}

pub fn record(entry: HistoryEntry) {
    let _ = store().add(entry);
}
