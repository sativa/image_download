//! Tauri command handlers for the mock backend.
//! Real implementations land in Plan B.

use serde::Serialize;

#[derive(Serialize)]
pub struct StartDownloadResp {
    pub download_id: String,
}

#[derive(Debug, Serialize)]
pub struct EstimateOutput {
    pub tile_count: u32,
    pub pixel_w: u32,
    pub pixel_h: u32,
    pub est_size_mb: f64,
    pub est_seconds: f64,
}

#[tauri::command]
pub async fn estimate_output(
    bbox: [f64; 4],
    zoom: u32,
    source: String,
) -> Result<EstimateOutput, String> {
    if zoom < 8 || zoom > 23 {
        return Err(format!("zoom {} out of range 8..23", zoom));
    }
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        return Err("invalid bbox".into());
    }
    // Web-mercator tile math.
    let n = 2_f64.powi(zoom as i32);
    let lon_w = bbox[0];
    let lon_e = bbox[2];
    let lat_s = bbox[1].max(-85.0511);
    let lat_n = bbox[3].min(85.0511);
    let x0 = ((lon_w + 180.0) / 360.0 * n).floor() as i64;
    let x1 = ((lon_e + 180.0) / 360.0 * n).ceil() as i64;
    let y0 = ((1.0
        - (lat_n.to_radians().tan() + 1.0 / lat_n.to_radians().cos()).ln()
            / std::f64::consts::PI)
        / 2.0
        * n)
        .floor() as i64;
    let y1 = ((1.0
        - (lat_s.to_radians().tan() + 1.0 / lat_s.to_radians().cos()).ln()
            / std::f64::consts::PI)
        / 2.0
        * n)
        .ceil() as i64;
    let tx = (x1 - x0).max(1) as u32;
    let ty = (y1 - y0).max(1) as u32;
    let tile_count = tx * ty;
    let pixel_w = tx * 256;
    let pixel_h = ty * 256;
    // Heuristic: 30 KB/tile JPEG, 50 tiles/sec.
    let est_size_mb = tile_count as f64 * 30.0 / 1024.0;
    let est_seconds = (tile_count as f64 / 50.0).max(1.0);
    let _ = source; // accepted but ignored in mock
    Ok(EstimateOutput {
        tile_count,
        pixel_w,
        pixel_h,
        est_size_mb,
        est_seconds,
    })
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
