//! Vector file parsing command. Wraps `core::vector::parse_vector`.

use crate::core::vector::parse_vector;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, Serialize)]
pub struct ParseVectorResp {
    pub bbox: [f64; 4],
    pub geometry: serde_json::Value,
    pub layer_count: u32,
}

#[tauri::command]
pub fn parse_vector_file(path: String) -> Result<ParseVectorResp, String> {
    let parsed = parse_vector(&PathBuf::from(&path)).map_err(|e| e.to_string())?;
    let geometry = serde_json::to_value(&parsed.geometry).map_err(|e| e.to_string())?;
    Ok(ParseVectorResp {
        bbox: parsed.bbox,
        geometry,
        layer_count: parsed.layer_count,
    })
}
