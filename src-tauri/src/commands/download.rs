//! Download pipeline commands: estimate, start, cancel, retry.
//! Wires core::{tiles, sources, downloader, stitcher, cog} into Tauri events.

use crate::commands::history::record;
use crate::commands::runner::Runner;
use crate::core::cog::{bbox_3857_from_range, write_cog, write_preview_png, CogParams};
use crate::core::downloader::{download_all, DownloadConfig, ProgressUpdate};
use crate::core::history::HistoryEntry;
use crate::core::sources::SourceKind;
use crate::core::stitcher::stitch_rgba;
use crate::core::tiles::range_for_bbox;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};
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
    if !(8..=23).contains(&zoom) {
        return Err(format!("zoom {} out of range 8..23", zoom));
    }
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        return Err("invalid bbox".into());
    }
    let _ = source;
    let range = range_for_bbox(bbox, zoom);
    let tile_count = range.count() as u32;
    let tx = (range.x_max - range.x_min + 1) as u32;
    let ty = (range.y_max - range.y_min + 1) as u32;
    Ok(EstimateOutput {
        tile_count,
        pixel_w: tx * 256,
        pixel_h: ty * 256,
        est_size_mb: tile_count as f64 * 30.0 / 1024.0,
        est_seconds: (tile_count as f64 / 50.0).max(1.0),
    })
}

#[derive(Debug, Deserialize, Clone)]
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
struct TileFailedEvent {
    download_id: String,
    x: i64,
    y: i64,
    z: u32,
    attempt: u32,
    error: String,
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

#[tauri::command]
pub async fn start_download(
    app: AppHandle,
    runner: State<'_, Runner>,
    args: StartDownloadArgs,
) -> Result<StartDownloadResp, String> {
    if args.bbox[0] >= args.bbox[2] || args.bbox[1] >= args.bbox[3] {
        return Err("invalid bbox".into());
    }
    let source_kind = SourceKind::parse(&args.source)
        .or_else(|| (args.source == "auto").then_some(SourceKind::Esri))
        .ok_or_else(|| format!("unknown source: {}", args.source))?;

    let id = Uuid::new_v4().to_string();
    let token = runner.register(id.clone());
    runner.stash_args(id.clone(), args.clone());

    let app_clone = app.clone();
    let id_clone = id.clone();
    let args_clone = args.clone();

    tokio::spawn(async move {
        run_pipeline(app_clone, id_clone, args_clone, source_kind, token).await;
    });

    Ok(StartDownloadResp { download_id: id })
}

async fn run_pipeline(
    app: AppHandle,
    id: String,
    args: StartDownloadArgs,
    source: SourceKind,
    token: tokio_util::sync::CancellationToken,
) {
    let start = Instant::now();
    let _ = app.emit(
        "download://stage",
        StageEvent {
            download_id: id.clone(),
            stage: "downloading".into(),
        },
    );

    let range = range_for_bbox(args.bbox, args.zoom);
    let coords: Vec<_> = range.iter().collect();
    let total = coords.len() as u32;

    let cfg = DownloadConfig {
        max_retries: args.retry_per_tile,
        backoff_base: Duration::from_millis(200),
        timeout_per_request: Duration::from_secs(30),
    };

    // Throttle progress to ~4 Hz; spec §3.2.
    let last_emit = Arc::new(std::sync::Mutex::new(
        Instant::now() - Duration::from_secs(1),
    ));
    let app_for_progress = app.clone();
    let id_for_progress = id.clone();
    let progress_cb = move |p: ProgressUpdate| {
        if let Some(c) = p.last_failed {
            let _ = app_for_progress.emit(
                "download://tile-failed",
                TileFailedEvent {
                    download_id: id_for_progress.clone(),
                    x: c.x,
                    y: c.y,
                    z: c.z,
                    attempt: 0,
                    error: "exhausted".into(),
                },
            );
        }
        let mut last = last_emit.lock().unwrap();
        let now = Instant::now();
        let force = p.completed == p.total;
        if !force && now.duration_since(*last) < Duration::from_millis(250) {
            return;
        }
        *last = now;
        let elapsed = (now - start).as_secs_f64();
        let speed = (p.bytes_downloaded as f64 / 1.0e6) / elapsed.max(0.01);
        let eta = if p.completed >= p.total {
            0.0
        } else {
            elapsed * (p.total - p.completed) as f64 / p.completed.max(1) as f64
        };
        let _ = app_for_progress.emit(
            "download://progress",
            ProgressEvent {
                download_id: id_for_progress.clone(),
                completed: p.completed,
                total: p.total,
                bytes_downloaded: p.bytes_downloaded,
                current_speed_mbps: speed,
                elapsed_sec: elapsed,
                eta_sec: eta,
            },
        );
    };

    let downloaded = download_all(
        coords,
        source,
        cfg,
        args.max_concurrency.max(1) as usize,
        token.clone(),
        progress_cb,
    )
    .await;

    if token.is_cancelled() {
        let _ = app.emit(
            "download://done",
            DoneEvent::Err {
                download_id: id,
                ok: false,
                error: "cancelled".into(),
            },
        );
        return;
    }

    let failed_tiles = downloaded.iter().filter(|t| t.bytes.is_none()).count() as u32;

    let _ = app.emit(
        "download://stage",
        StageEvent {
            download_id: id.clone(),
            stage: "stitching".into(),
        },
    );
    let img = stitch_rgba(&downloaded, range);

    let _ = app.emit(
        "download://stage",
        StageEvent {
            download_id: id.clone(),
            stage: "writing_cog".into(),
        },
    );
    let bbox_3857 = bbox_3857_from_range(range);
    let out_path = PathBuf::from(&args.output_path);
    if let Err(e) = write_cog(
        &img,
        &CogParams {
            bbox_3857,
            zoom: args.zoom,
        },
        &out_path,
    ) {
        let _ = app.emit(
            "download://done",
            DoneEvent::Err {
                download_id: id,
                ok: false,
                error: format!("write_cog failed: {e}"),
            },
        );
        return;
    }

    let preview_path = if args.write_preview_png {
        let _ = app.emit(
            "download://stage",
            StageEvent {
                download_id: id.clone(),
                stage: "writing_preview".into(),
            },
        );
        let pp = out_path.with_extension("preview.png");
        write_preview_png(&img, &pp, 1024)
            .ok()
            .map(|_| pp.to_string_lossy().into_owned())
    } else {
        None
    };

    let duration_sec = start.elapsed().as_secs_f64();
    let output_size_mb = std::fs::metadata(&out_path)
        .map(|m| m.len() as f64 / 1.0e6)
        .unwrap_or(0.0);

    record(HistoryEntry {
        bbox: args.bbox,
        zoom: args.zoom,
        source: args.source.clone(),
        output_path: args.output_path.clone(),
        ok: failed_tiles == 0,
        duration_sec,
        total_tiles: total,
        failed_tiles,
        output_size_mb,
        finished_at: format!(
            "epoch:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ),
    });

    let _ = app.emit(
        "download://done",
        DoneEvent::Ok {
            download_id: id,
            ok: true,
            output_path: args.output_path.clone(),
            preview_path,
            bbox: args.bbox,
            zoom: args.zoom,
            source_used: source.as_str().into(),
            duration_sec,
            total_tiles: total,
            failed_tiles,
            output_size_mb,
        },
    );
}

#[tauri::command]
pub fn cancel_download(
    runner: State<'_, Runner>,
    download_id: String,
) -> Result<serde_json::Value, String> {
    if !runner.cancel(&download_id) {
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
    let Some(args) = runner.lookup_args(&download_id) else {
        return Err(format!("no args stashed for {}", download_id));
    };
    let source_kind = SourceKind::parse(&args.source).unwrap_or(SourceKind::Esri);
    let token = runner.register(download_id.clone());
    let app_clone = app.clone();
    let id_clone = download_id.clone();
    tokio::spawn(async move {
        run_pipeline(app_clone, id_clone, args, source_kind, token).await;
    });
    Ok(serde_json::json!({ "ok": true }))
}
