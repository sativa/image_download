//! Tauri command handlers for the mock backend.
//! Plan B will replace this entire module with real implementations.

use crate::history::HistoryEntry;
use crate::mocks::history_commands::record;
use crate::mocks::runner::Runner;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};
use tokio::time::sleep;
use uuid::Uuid;

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
    let est_size_mb = tile_count as f64 * 30.0 / 1024.0;
    let est_seconds = (tile_count as f64 / 50.0).max(1.0);
    let _ = source;
    Ok(EstimateOutput {
        tile_count,
        pixel_w,
        pixel_h,
        est_size_mb,
        est_seconds,
    })
}

#[derive(Debug, Deserialize)]
pub struct StartDownloadArgs {
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    pub output_path: String,
    pub max_concurrency: u32,
    pub retry_per_tile: u32,
    pub write_preview_png: bool,
}

#[derive(Debug, Serialize)]
pub struct StartDownloadResp {
    pub download_id: String,
}

#[derive(Debug, Serialize, Clone)]
struct ProgressEvent {
    download_id: String,
    completed: u32,
    total: u32,
    bytes_downloaded: u64,
    current_speed_mbps: f64,
    elapsed_sec: f64,
    eta_sec: f64,
}

#[derive(Debug, Serialize, Clone)]
struct StageEvent {
    download_id: String,
    stage: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum DoneEvent {
    Ok {
        download_id: String,
        ok: bool,
        output_path: String,
        preview_path: Option<String>,
        bbox: [f64; 4],
        zoom: u32,
        source_used: String,
        duration_sec: f64,
        total_tiles: u32,
        failed_tiles: u32,
        output_size_mb: f64,
    },
    Err {
        download_id: String,
        ok: bool,
        error: String,
    },
}

const MOCK_TOTAL: u32 = 100;
const MOCK_DURATION_SEC: f64 = 5.0;
const TICKS: u32 = 50;

#[tauri::command]
pub async fn start_download(
    app: AppHandle,
    runner: State<'_, Runner>,
    args: StartDownloadArgs,
) -> Result<StartDownloadResp, String> {
    if args.bbox[0] >= args.bbox[2] || args.bbox[1] >= args.bbox[3] {
        return Err("invalid bbox".into());
    }
    let id = Uuid::new_v4().to_string();
    let token = runner.register(id.clone());
    let app_clone = app.clone();
    let id_clone = id.clone();
    let bbox = args.bbox;
    let zoom = args.zoom;
    let source = args.source.clone();
    let output_path = args.output_path.clone();
    let write_preview = args.write_preview_png;

    tokio::spawn(async move {
        let _ = app_clone.emit(
            "download://stage",
            StageEvent {
                download_id: id_clone.clone(),
                stage: "downloading".into(),
            },
        );
        let start = Instant::now();
        let tick_dur = Duration::from_secs_f64(MOCK_DURATION_SEC / TICKS as f64);
        let bytes_per_tile = 30_000u64;
        for i in 1..=TICKS {
            tokio::select! {
                _ = sleep(tick_dur) => {},
                _ = token.cancelled() => {
                    let _ = app_clone.emit(
                        "download://done",
                        DoneEvent::Err {
                            download_id: id_clone.clone(),
                            ok: false,
                            error: "cancelled".into(),
                        },
                    );
                    return;
                }
            }
            let completed = (i * MOCK_TOTAL) / TICKS;
            let elapsed = start.elapsed().as_secs_f64();
            let eta = if i == TICKS {
                0.0
            } else {
                (MOCK_DURATION_SEC - elapsed).max(0.0)
            };
            let bytes = bytes_per_tile * completed as u64;
            let speed = (bytes as f64 / 1.0e6) / elapsed.max(0.01);
            let _ = app_clone.emit(
                "download://progress",
                ProgressEvent {
                    download_id: id_clone.clone(),
                    completed,
                    total: MOCK_TOTAL,
                    bytes_downloaded: bytes,
                    current_speed_mbps: speed,
                    elapsed_sec: elapsed,
                    eta_sec: eta,
                },
            );
        }
        let _ = app_clone.emit(
            "download://stage",
            StageEvent {
                download_id: id_clone.clone(),
                stage: "stitching".into(),
            },
        );
        sleep(Duration::from_millis(300)).await;
        let _ = app_clone.emit(
            "download://stage",
            StageEvent {
                download_id: id_clone.clone(),
                stage: "writing_cog".into(),
            },
        );
        sleep(Duration::from_millis(400)).await;
        if write_preview {
            let _ = app_clone.emit(
                "download://stage",
                StageEvent {
                    download_id: id_clone.clone(),
                    stage: "writing_preview".into(),
                },
            );
            sleep(Duration::from_millis(200)).await;
        }
        let total_dur = start.elapsed().as_secs_f64();
        let preview_path = if write_preview {
            Some(format!(
                "{}.preview.png",
                output_path.trim_end_matches(".tif")
            ))
        } else {
            None
        };
        let entry = HistoryEntry {
            bbox,
            zoom,
            source: source.clone(),
            output_path: output_path.clone(),
            ok: true,
            duration_sec: total_dur,
            total_tiles: MOCK_TOTAL,
            failed_tiles: 0,
            output_size_mb: (bytes_per_tile * MOCK_TOTAL as u64) as f64 / 1.0e6,
            finished_at: chrono_iso_now(),
        };
        record(entry);
        let _ = app_clone.emit(
            "download://done",
            DoneEvent::Ok {
                download_id: id_clone,
                ok: true,
                output_path,
                preview_path,
                bbox,
                zoom,
                source_used: source,
                duration_sec: total_dur,
                total_tiles: MOCK_TOTAL,
                failed_tiles: 0,
                output_size_mb: (bytes_per_tile * MOCK_TOTAL as u64) as f64 / 1.0e6,
            },
        );
    });

    Ok(StartDownloadResp { download_id: id })
}

#[tauri::command]
pub fn cancel_download(
    runner: State<'_, Runner>,
    download_id: String,
) -> Result<serde_json::Value, String> {
    let cancelled = runner.cancel(&download_id);
    if !cancelled {
        return Err(format!("unknown download_id: {}", download_id));
    }
    Ok(serde_json::json!({ "ok": true }))
}

#[tauri::command]
pub async fn retry_failed(
    app: AppHandle,
    runner: State<'_, Runner>,
    download_id: String,
) -> Result<serde_json::Value, String> {
    let token = runner.register(download_id.clone());
    let id_clone = download_id.clone();
    let app_clone = app.clone();
    tokio::spawn(async move {
        let _ = app_clone.emit(
            "download://stage",
            StageEvent {
                download_id: id_clone.clone(),
                stage: "downloading".into(),
            },
        );
        for i in 1..=10u32 {
            tokio::select! {
                _ = sleep(Duration::from_millis(100)) => {},
                _ = token.cancelled() => {
                    let _ = app_clone.emit(
                        "download://done",
                        DoneEvent::Err {
                            download_id: id_clone.clone(),
                            ok: false,
                            error: "cancelled".into(),
                        },
                    );
                    return;
                }
            }
            let _ = app_clone.emit(
                "download://progress",
                ProgressEvent {
                    download_id: id_clone.clone(),
                    completed: i * 10,
                    total: MOCK_TOTAL,
                    bytes_downloaded: 0,
                    current_speed_mbps: 1.0,
                    elapsed_sec: i as f64 * 0.1,
                    eta_sec: (10 - i) as f64 * 0.1,
                },
            );
        }
        let _ = app_clone.emit(
            "download://done",
            DoneEvent::Ok {
                download_id: id_clone,
                ok: true,
                output_path: String::new(),
                preview_path: None,
                bbox: [0.0; 4],
                zoom: 0,
                source_used: String::new(),
                duration_sec: 1.0,
                total_tiles: MOCK_TOTAL,
                failed_tiles: 0,
                output_size_mb: 0.0,
            },
        );
    });
    Ok(serde_json::json!({ "ok": true }))
}

#[tauri::command]
pub fn parse_vector_file(_path: String) -> Result<serde_json::Value, String> {
    Err("vector parsing pending Plan A".into())
}

fn chrono_iso_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now();
    let epoch = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("epoch:{}", epoch)
}
