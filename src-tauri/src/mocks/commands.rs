//! Tauri command handlers for the mock backend.
//! Real implementations land in Plan B.

use serde::Serialize;

#[derive(Serialize)]
pub struct StartDownloadResp {
    pub download_id: String,
}

#[tauri::command]
pub async fn estimate_output() -> Result<(), String> {
    Err("not implemented yet (Task 7.1)".into())
}

#[tauri::command]
pub async fn start_download() -> Result<StartDownloadResp, String> {
    Err("not implemented yet (Task 7.2)".into())
}

#[tauri::command]
pub async fn cancel_download() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 7.3)".into())
}

#[tauri::command]
pub async fn retry_failed() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 7.4)".into())
}

#[tauri::command]
pub async fn parse_vector_file() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 4.2)".into())
}
