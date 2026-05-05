//! Download pipeline commands: estimate, start, cancel, retry.
//! Wires core::{tiles, sources, downloader, stitcher, cog} into Tauri events.

use crate::commands::history::record;
use crate::commands::runner::Runner;
use crate::core::cog::{bbox_3857_from_range, write_cog, write_preview_png, CogParams};
use crate::core::downloader::{download_all_with_sink, DownloadConfig, DownloadedTile, ProgressUpdate};
use crate::core::history::HistoryEntry;
use crate::core::job::Job;
use crate::core::sources::{pick_auto, SourceKind};
use crate::core::stitcher::stitch_rgba_with_progress;
use crate::core::tiles::range_for_bbox;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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
    // "auto" → probe both Esri/Google and use the faster one. Other values
    // ("esri", "google") parse to a concrete kind directly.
    let source_kind = if args.source == "auto" {
        pick_auto().await
    } else {
        SourceKind::parse(&args.source)
            .ok_or_else(|| format!("unknown source: {}", args.source))?
    };

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

    // ── Resume support ───────────────────────────────────────────────────
    // Open or create a Job in the user-chosen output folder. The Job stem
    // is derived from bbox+zoom+source so re-running with the same params
    // resumes from the WAL; changing any param starts a fresh job.
    let mut out_dir = PathBuf::from(&args.output_path);
    // Defensive: history rows stored the .tif file path as `output_path`,
    // and earlier UI builds restored that directly into input.outputPath.
    // If we receive something that points at (or names) a file, treat its
    // parent as the folder so we don't create a nested `<stem>.tif/`.
    if out_dir.is_file()
        || out_dir
            .extension()
            .map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
            .unwrap_or(false)
    {
        if let Some(parent) = out_dir.parent() {
            log::warn!(
                "output_path looks like a file ({}); using parent {} as the folder",
                out_dir.display(),
                parent.display()
            );
            out_dir = parent.to_path_buf();
        }
    }
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        let _ = app.emit(
            "download://done",
            DoneEvent::Err {
                download_id: id,
                ok: false,
                error: format!("create output dir failed: {e}"),
            },
        );
        return;
    }
    // Canonical "user-chosen folder" string for every history row this run
    // writes — so a future restore feeds it back unchanged into the picker.
    let out_dir_str = out_dir.to_string_lossy().into_owned();
    let job = match Job::open_or_create(&out_dir, args.bbox, args.zoom, &args.source, total) {
        Ok(j) => j,
        Err(e) => {
            let _ = app.emit(
                "download://done",
                DoneEvent::Err {
                    download_id: id,
                    ok: false,
                    error: format!("init job state failed: {e}"),
                },
            );
            return;
        }
    };
    let completed_set = job.load_completed().unwrap_or_default();
    let already_done = completed_set.len() as u32;
    if already_done > 0 {
        let _ = app.emit(
            "download://stage",
            StageEvent {
                download_id: id.clone(),
                stage: format!("resuming ({}/{} tiles already cached)", already_done, total),
            },
        );
    }

    // Record an in-progress entry early so the user can see the task in
    // History even if the app crashes mid-download. Subsequent record calls
    // on the same (bbox, zoom, source) replace this row (Store dedupes).
    record_history_entry(
        &args,
        out_dir_str.clone(),
        total,
        already_done,
        0,
        0.0,
        0.0,
        HistoryEntry::STATUS_IN_PROGRESS,
    );

    // Partition: cached tiles bypass the network; missing or corrupt cache
    // entries fall back to a fresh download.
    let mut cached_tiles: Vec<DownloadedTile> = Vec::with_capacity(already_done as usize);
    let mut todo: Vec<_> = Vec::with_capacity(coords.len() - already_done as usize);
    for c in coords {
        if completed_set.contains(&(c.x, c.y)) {
            match job.load_tile(c) {
                Ok(b) => cached_tiles.push(DownloadedTile {
                    coord: c,
                    bytes: Some(b),
                }),
                Err(_) => todo.push(c), // cache file missing → redownload
            }
        } else {
            todo.push(c);
        }
    }

    let cfg = DownloadConfig {
        max_retries: args.retry_per_tile,
        backoff_base: Duration::from_millis(200),
        timeout_per_request: Duration::from_secs(30),
    };

    // Throttle progress to ~4 Hz; spec §3.2.
    let last_emit = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(1)));
    // Throttle history-row writes to once every 5 seconds so the in_progress
    // entry's `completed_tiles` stays roughly fresh without hammering disk
    // (`Store::add` fsyncs the whole history.json on every call).
    let last_history_write = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(60)));
    let app_for_progress = app.clone();
    let id_for_progress = id.clone();
    let total_for_progress = total;
    let already_for_progress = already_done;
    let args_for_history = args.clone();
    let out_dir_for_history = out_dir_str.clone();
    let start_for_history = start;
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
        // Project download_all's local counts (todo-only) onto the *whole* job
        // so the UI sees a single 0..total progress bar.
        let completed_overall = already_for_progress + p.completed;
        let mut last = last_emit.lock().unwrap();
        let now = Instant::now();
        let force = completed_overall == total_for_progress;
        if !force && now.duration_since(*last) < Duration::from_millis(250) {
            return;
        }
        *last = now;
        let elapsed = (now - start).as_secs_f64();
        let speed = (p.bytes_downloaded as f64 / 1.0e6) / elapsed.max(0.01);
        let eta = if completed_overall >= total_for_progress {
            0.0
        } else {
            let remaining = (total_for_progress - completed_overall) as f64;
            // Use freshly-downloaded count for the rate estimate, not overall:
            // cached tiles took ~0s to "complete" and would skew ETA towards 0.
            elapsed * remaining / p.completed.max(1) as f64
        };
        let _ = app_for_progress.emit(
            "download://progress",
            ProgressEvent {
                download_id: id_for_progress.clone(),
                completed: completed_overall,
                total: total_for_progress,
                bytes_downloaded: p.bytes_downloaded,
                current_speed_mbps: speed,
                elapsed_sec: elapsed,
                eta_sec: eta,
            },
        );

        // Periodically refresh the in_progress history row so a hard crash
        // (or kill -9) leaves a row that reflects roughly where we got, not
        // the stale `already_done` snapshot from task startup.
        let mut last_h = last_history_write.lock().unwrap();
        if now.duration_since(*last_h) >= Duration::from_secs(5) {
            *last_h = now;
            record_history_entry(
                &args_for_history,
                out_dir_for_history.clone(),
                total_for_progress,
                completed_overall,
                0,
                start_for_history.elapsed().as_secs_f64(),
                0.0,
                HistoryEntry::STATUS_IN_PROGRESS,
            );
        }
    };

    // Per-tile sink: persist bytes to <stem>.tiles/{x}_{y}.bin and append the
    // coord to the WAL. Sync I/O — runs on the channel-receiver task.
    let job_for_sink = job.clone();
    let sink = move |c, b: &bytes::Bytes| {
        if let Err(e) = job_for_sink.record_tile(c, b) {
            log::warn!("record_tile failed: {e}"); // resume will retry this tile next run
        }
    };

    let downloaded = download_all_with_sink(
        todo,
        source,
        cfg,
        args.max_concurrency.max(1) as usize,
        token.clone(),
        progress_cb,
        sink,
    )
    .await;

    if token.is_cancelled() {
        // Recompute "how far did we get" from the WAL — that's the source of
        // truth, not in-memory `downloaded` (which may have aborted entries).
        let cancelled_completed = job.load_completed().map(|s| s.len() as u32).unwrap_or(0);
        record_history_entry(
            &args,
            out_dir_str.clone(),
            total,
            cancelled_completed,
            0,
            start.elapsed().as_secs_f64(),
            0.0,
            HistoryEntry::STATUS_CANCELLED,
        );
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

    // Merge cached + freshly downloaded for stitching.
    let mut all_tiles = cached_tiles;
    all_tiles.extend(downloaded);
    let failed_tiles = all_tiles.iter().filter(|t| t.bytes.is_none()).count() as u32;

    let _ = app.emit(
        "download://stage",
        StageEvent {
            download_id: id.clone(),
            stage: "stitching".into(),
        },
    );
    // Parallel decode + serial composite. The closure runs from rayon worker
    // threads; tauri::AppHandle::emit is Send + thread-safe, so we just clone
    // what's needed and fire IPC events at ~4 Hz from inside.
    let app_for_stitch = app.clone();
    let id_for_stitch = id.clone();
    let stitch_started = Instant::now();
    let img = stitch_rgba_with_progress(&all_tiles, range, move |done, total| {
        let _ = app_for_stitch.emit(
            "download://progress",
            ProgressEvent {
                download_id: id_for_stitch.clone(),
                completed: done,
                total,
                bytes_downloaded: 0,
                current_speed_mbps: 0.0,
                elapsed_sec: stitch_started.elapsed().as_secs_f64(),
                eta_sec: 0.0,
            },
        );
    });

    let _ = app.emit(
        "download://stage",
        StageEvent {
            download_id: id.clone(),
            stage: "writing_cog".into(),
        },
    );
    let bbox_3857 = bbox_3857_from_range(range);
    let out_path = job.output_tif_path();
    if let Err(e) = write_cog(
        &img,
        &CogParams {
            bbox_3857,
            zoom: args.zoom,
        },
        &out_path,
    ) {
        // NB: do NOT clear job state on COG failure — keeping the cache lets
        // the next run skip straight to stitching with the same tiles.
        // all_tiles already merges cached + freshly downloaded; cached always
        // have Some(bytes), so filter(is_some) is the correct overall count.
        let completed_overall = all_tiles.iter().filter(|t| t.bytes.is_some()).count() as u32;
        record_history_entry(
            &args,
            out_dir_str.clone(),
            total,
            completed_overall,
            failed_tiles,
            start.elapsed().as_secs_f64(),
            0.0,
            HistoryEntry::STATUS_FAILED,
        );
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

    // COG written; the job's tiles cache and WAL are no longer needed.
    if let Err(e) = job.clear_state() {
        log::warn!("clear job state failed: {e}");
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

    let written_path = out_path.to_string_lossy().into_owned();
    record_history_entry(
        &args,
        out_dir_str.clone(),
        total,
        total - failed_tiles,
        failed_tiles,
        duration_sec,
        output_size_mb,
        HistoryEntry::STATUS_COMPLETED,
    );

    let _ = app.emit(
        "download://done",
        DoneEvent::Ok {
            download_id: id,
            ok: true,
            output_path: written_path,
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
    // Honour "auto" on retry too: re-probe in case network conditions have
    // shifted since the original run. Falls back to Esri only if the stashed
    // value is something we don't recognise (defensive).
    let source_kind = if args.source == "auto" {
        pick_auto().await
    } else {
        SourceKind::parse(&args.source).unwrap_or(SourceKind::Esri)
    };
    let token = runner.register(download_id.clone());
    let app_clone = app.clone();
    let id_clone = download_id.clone();
    tokio::spawn(async move {
        run_pipeline(app_clone, id_clone, args, source_kind, token).await;
    });
    Ok(serde_json::json!({ "ok": true }))
}

/// Persist a HistoryEntry for one lifecycle event (in_progress / cancelled /
/// failed / completed). All four entry points in `run_pipeline` go through
/// this so the field set stays consistent.
///
/// `output_path` should be the *final* COG path even before the file exists,
/// so the user can identify the entry by where its tiles will/do live.
fn record_history_entry(
    args: &StartDownloadArgs,
    output_path: String,
    total_tiles: u32,
    completed_tiles: u32,
    failed_tiles: u32,
    duration_sec: f64,
    output_size_mb: f64,
    status: &str,
) {
    record(HistoryEntry {
        bbox: args.bbox,
        zoom: args.zoom,
        source: args.source.clone(),
        output_path,
        ok: status == HistoryEntry::STATUS_COMPLETED && failed_tiles == 0,
        duration_sec,
        total_tiles,
        failed_tiles,
        output_size_mb,
        finished_at: format!(
            "epoch:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ),
        completed_tiles,
        status: status.to_string(),
    });
}
