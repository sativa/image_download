//! Land-cover classification command.
//!
//! Spawns the Python sidecar (`sidecar/sam3_classify/__main__.py`) as a
//! subprocess and relays its NDJSON stdout stream as Tauri events. The
//! sidecar reads a GeoTIFF, runs SAM 3 with the prompts in
//! `sam3_classify/prompts.py`, and writes `<stem>.landform.tif` plus
//! `<stem>.landform.legend.json`. No frontend code knows that Python is
//! involved — the IPC contract is just `landform://{stage, progress, done}`.
//!
//! Lifecycle:
//!   start_classify(args) → spawns child, returns classify_id immediately
//!   ├─ stdout lines → re-broadcast as events keyed by classify_id
//!   ├─ stderr lines → log only (sidecar uses stderr for free-form logs)
//!   └─ child exit  → emit landform://done if not already emitted
//!   cancel_classify(id) → cancellation token aborts wait + kills child

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Default)]
pub struct ClassifyRunner {
    tokens: Mutex<std::collections::HashMap<String, CancellationToken>>,
}

impl ClassifyRunner {
    pub fn register(&self, id: String) -> CancellationToken {
        let t = CancellationToken::new();
        self.tokens.lock().unwrap().insert(id, t.clone());
        t
    }
    pub fn cancel(&self, id: &str) -> bool {
        if let Some(t) = self.tokens.lock().unwrap().remove(id) {
            t.cancel();
            true
        } else {
            false
        }
    }
    pub fn forget(&self, id: &str) {
        self.tokens.lock().unwrap().remove(id);
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct StartClassifyArgs {
    /// Input GeoTIFF produced by the downloader.
    pub input_path: String,
    /// Folder for output. Defaults to the input's parent if omitted.
    pub output_dir: Option<String>,
    /// "auto" | "cpu" | "mps" | "cuda". "auto" lets the sidecar pick.
    pub device: Option<String>,
    /// SAM 3 confidence threshold; tighter = fewer false detections.
    pub confidence: Option<f32>,
    /// "slic" | "sam3" | "dino". Defaults to "slic" — bench_compare.py
    /// on 3 real scenes showed slic has the best boundary-alignment
    /// score (1.725 vs 1.565 sam3 vs 1.391 dino) while being 3-4× faster
    /// than sam3. Switch to sam3 when you specifically need object-aware
    /// boundaries (e.g. building outlines); dino is for unsupervised
    /// feature clustering, not the current colour-rule pipeline.
    pub backend: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StartClassifyResp {
    pub classify_id: String,
}

#[derive(Debug, Serialize, Clone)]
struct StageEvent {
    classify_id: String,
    stage: String,
}

#[derive(Debug, Serialize, Clone)]
struct ProgressEvent {
    classify_id: String,
    done: u32,
    total: u32,
    current_prompt: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum DoneEvent {
    Ok {
        classify_id: String,
        ok: bool,
        /// Canonical product: GeoPackage in source CRS (EPSG:3857).
        label_gpkg: String,
        /// Same geometry reprojected to WGS84 GeoJSON for map display.
        overlay_geojson: String,
        legend_json: String,
        /// [minLon, minLat, maxLon, maxLat] in WGS84 of the COG's
        /// tile-snapped extent. The GeoJSON features already carry this
        /// implicitly, but emitting it explicitly lets the map fit the
        /// camera before the FeatureCollection is fetched.
        overlay_bbox_wgs84: Option<[f64; 4]>,
        duration_sec: f64,
        stats: serde_json::Value,
    },
    Err {
        classify_id: String,
        ok: bool,
        error: String,
    },
}

/// Defaults are hard-coded for the current development machine. Each one
/// can be overridden by the corresponding environment variable so the
/// same binary works for a teammate who installed everything elsewhere.
fn default_python() -> String {
    // The py3.12 venv is the one set up for samgeo3's auto-mask-generator
    // path. Falling back to the older py3.14 venv breaks the new pipeline
    // because samgeo's inst_interactivity requires torch.jit.script to
    // work end-to-end, which fails on 3.14.
    std::env::var("SAM3_PYTHON")
        .unwrap_or_else(|_| "/Users/zhangfeng/D/sam3/sam3_env_py312/bin/python".into())
}
fn default_sidecar_dir() -> PathBuf {
    std::env::var("SAM3_SIDECAR_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
        })
}
fn default_weights() -> String {
    std::env::var("SAM3_WEIGHTS")
        .unwrap_or_else(|_| "/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt".into())
}

/// Trained DINOv3-Sat + FreqFusion + GDLX binary-cropland checkpoint.
fn default_cropland_weights() -> String {
    std::env::var("CROPLAND_WEIGHTS")
        .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/cropland_gdlxff.pt".into())
}

/// Trained DINOv3-Sat 8-class land-cover + parcel-boundary-head checkpoint
/// (used by both `landcover` and `parcel_bh` backends).
fn default_landcover_weights() -> String {
    std::env::var("LANDCOVER_WEIGHTS")
        .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/landcover8_bh.pt".into())
}

/// DINOv3-Sat backbone dir the cropland checkpoint is loaded on top of.
fn default_backbone_dir() -> String {
    std::env::var("CROPLAND_BACKBONE")
        .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/dinov3-vitl16-sat493m".into())
}

/// Map an input TIFF path to its landform output path. Lives alongside
/// the input so the existing job-state cleanup never touches it.
fn derive_output_path(input: &str, override_dir: Option<&str>) -> PathBuf {
    let input_path = PathBuf::from(input);
    let stem = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let parent = override_dir
        .map(PathBuf::from)
        .or_else(|| input_path.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    parent.join(format!("{stem}.landform.tif"))
}

#[tauri::command]
pub async fn start_classify(
    app: AppHandle,
    runner: State<'_, ClassifyRunner>,
    args: StartClassifyArgs,
) -> Result<StartClassifyResp, String> {
    if !PathBuf::from(&args.input_path).exists() {
        return Err(format!("input not found: {}", args.input_path));
    }
    let id = Uuid::new_v4().to_string();
    let token = runner.register(id.clone());

    let app_clone = app.clone();
    let id_clone = id.clone();
    tokio::spawn(async move {
        run_sidecar(app_clone, id_clone, args, token).await;
    });

    Ok(StartClassifyResp { classify_id: id })
}

#[tauri::command]
pub fn cancel_classify(runner: State<'_, ClassifyRunner>, classify_id: String) -> bool {
    runner.cancel(&classify_id)
}

/// Look up the GeoTIFF a history row points at.
///
/// Some history rows store `output_path` as the user-chosen folder (newer
/// schema), others as the .tif file directly (legacy). For the former we
/// scan the folder for any `*.tif` that mentions the row's zoom and
/// source — both new (`imagery_z17_esri_*.tif`) and old (`merge_imagery_z17_esri_*.tif`)
/// naming schemes are matched in one pass.
#[tauri::command]
pub fn resolve_history_tif(
    output_path: String,
    zoom: u32,
    source: String,
) -> Result<Option<String>, String> {
    let p = PathBuf::from(&output_path);
    if p.is_file() {
        return Ok(Some(output_path));
    }
    if !p.is_dir() {
        return Ok(None);
    }
    let needle = format!("z{zoom}_{source}");
    let entries = std::fs::read_dir(&p).map_err(|e| e.to_string())?;
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let ext_ok = path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
            .unwrap_or(false);
        if !ext_ok {
            continue;
        }
        // Skip anything that's already a landform output — those are
        // single-band uint8 rasters and don't make sense to feed back in.
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.contains(".landform."))
            .unwrap_or(false)
        {
            continue;
        }
        if path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.contains(&needle))
            .unwrap_or(false)
        {
            candidates.push(path);
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    // Pick the most recently modified candidate — handles the case where
    // a re-download produced a new file alongside an older one.
    candidates.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
    });
    Ok(candidates
        .last()
        .map(|p| p.to_string_lossy().into_owned()))
}

async fn run_sidecar(
    app: AppHandle,
    id: String,
    args: StartClassifyArgs,
    token: CancellationToken,
) {
    let start = Instant::now();
    let python = default_python();
    let sidecar_dir = default_sidecar_dir();
    let output_path =
        derive_output_path(&args.input_path, args.output_dir.as_deref());
    // Default backend: trained DINOv3-Sat binary cropland model (ready + tested), full-coverage.
    // Switch to "landcover" (7-class per-parcel) once that checkpoint is trained + on disk.
    let backend = args.backend.unwrap_or_else(|| "cropland".into());
    let is_trained = matches!(backend.as_str(), "landcover" | "cropland" | "parcel" | "parcel_bh");
    let weights = match backend.as_str() {
        // landcover + parcel_bh both use the 8-class + boundary-head checkpoint.
        "landcover" | "parcel_bh" => default_landcover_weights(),
        // parcel = SAM3 instances + the DINOv3 cropland model (weights = cropland ckpt;
        // SAM3 checkpoint comes from the sidecar's --sam3-weights default).
        "cropland" | "parcel" => default_cropland_weights(),
        _ => default_weights(),
    };
    let backbone_dir = default_backbone_dir();
    // The trained models use standard transformer ops (well-covered on MPS); the legacy
    // SAM-3 path defaults to CPU because its video ops fall back mid-graph.
    let device = args
        .device
        .unwrap_or_else(|| if is_trained { "mps".into() } else { "cpu".into() });
    // For the trained models, confidence is the softmax probability threshold (argmax-based,
    // 0.5); for SAM-3 it is the detection confidence.
    let confidence = args.confidence.unwrap_or(if is_trained { 0.5 } else { 0.05 });

    let _ = app.emit(
        "landform://stage",
        StageEvent {
            classify_id: id.clone(),
            stage: "spawning_sidecar".into(),
        },
    );

    let mut cmd = Command::new(&python);
    cmd.current_dir(&sidecar_dir)
        // Always expose the MPS fallback for whatever falls through to
        // MPS in the future. Harmless on CPU and CUDA paths.
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        // Reduce import-time spam SAM 3's video modules emit when CUDA
        // isn't available; keeps the user's stderr log readable.
        .env("PYTHONWARNINGS", "ignore")
        .args([
            "-m",
            "sam3_classify",
            "--input",
            &args.input_path,
            "--output",
            output_path.to_str().unwrap_or_default(),
            "--weights",
            &weights,
            "--device",
            &device,
            "--confidence",
            &confidence.to_string(),
            "--backend",
            &backend,
            "--backbone-dir",
            &backbone_dir,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = app.emit(
                "landform://done",
                DoneEvent::Err {
                    classify_id: id.clone(),
                    ok: false,
                    error: format!("failed to spawn sidecar: {e} (python={python})"),
                },
            );
            return;
        }
    };

    let stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            let _ = app.emit(
                "landform://done",
                DoneEvent::Err {
                    classify_id: id.clone(),
                    ok: false,
                    error: "sidecar stdout pipe missing".into(),
                },
            );
            let _ = child.kill().await;
            return;
        }
    };
    let stderr = child.stderr.take();

    // Pipe stderr lines into the log; the user only sees them on failure.
    if let Some(stderr) = stderr {
        let id_for_log = id.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::warn!("[classify {}] {}", id_for_log, line);
            }
        });
    }

    let mut reader = BufReader::new(stdout).lines();
    let mut done_emitted = false;

    loop {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                let _ = child.kill().await;
                let _ = app.emit(
                    "landform://done",
                    DoneEvent::Err {
                        classify_id: id.clone(),
                        ok: false,
                        error: "cancelled".into(),
                    },
                );
                done_emitted = true;
                break;
            }
            line = reader.next_line() => {
                match line {
                    Ok(Some(line)) => {
                        if let Some(record) = parse_ndjson(&line) {
                            done_emitted |= dispatch(&app, &id, record);
                        } else {
                            log::warn!("[classify {}] unparseable stdout: {}", id, line);
                        }
                    }
                    Ok(None) => break, // EOF: child closed stdout
                    Err(e) => {
                        log::warn!("[classify {}] stdout read error: {}", id, e);
                        break;
                    }
                }
            }
        }
    }

    // Reap the child so the OS releases the slot.
    let exit_status = child.wait().await.ok();
    if !done_emitted {
        let elapsed = start.elapsed().as_secs_f64();
        let code = exit_status.and_then(|s| s.code()).unwrap_or(-1);
        let _ = app.emit(
            "landform://done",
            DoneEvent::Err {
                classify_id: id.clone(),
                ok: false,
                error: format!(
                    "sidecar exited (code {}) after {:.1}s without a 'done' record",
                    code, elapsed
                ),
            },
        );
    }
    app.state::<ClassifyRunner>().forget(&id);
    let _ = start; // suppress unused-when-no-error warning
}

/// Parse one stdout line as NDJSON. Returns None for empty / non-JSON
/// lines (the sidecar should not emit any, but `print` from a third-party
/// import could leak through).
fn parse_ndjson(line: &str) -> Option<serde_json::Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.starts_with('{') {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

/// Translate one NDJSON record into a Tauri event. Returns true iff a
/// terminal (done/error) event was emitted, so the caller can suppress
/// the "no done record" fallback.
fn dispatch(app: &AppHandle, id: &str, record: serde_json::Value) -> bool {
    let ty = record.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ty {
        "stage" => {
            let stage = record
                .get("stage")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let _ = app.emit(
                "landform://stage",
                StageEvent {
                    classify_id: id.to_string(),
                    stage,
                },
            );
            false
        }
        "progress" => {
            let done = record.get("done").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let total = record.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let current_prompt = record
                .get("current_prompt")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let _ = app.emit(
                "landform://progress",
                ProgressEvent {
                    classify_id: id.to_string(),
                    done,
                    total,
                    current_prompt,
                },
            );
            false
        }
        "done" => {
            let label_gpkg = record
                .get("label_gpkg")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let overlay_geojson = record
                .get("overlay_geojson")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let legend_json = record
                .get("legend_json")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let overlay_bbox_wgs84 = record.get("overlay_bbox_wgs84").and_then(|v| {
                let arr = v.as_array()?;
                if arr.len() != 4 {
                    return None;
                }
                let mut out = [0f64; 4];
                for (i, item) in arr.iter().enumerate() {
                    out[i] = item.as_f64()?;
                }
                Some(out)
            });
            let stats = record.get("stats").cloned().unwrap_or(serde_json::json!({}));
            let _ = app.emit(
                "landform://done",
                DoneEvent::Ok {
                    classify_id: id.to_string(),
                    ok: true,
                    label_gpkg,
                    overlay_geojson,
                    legend_json,
                    overlay_bbox_wgs84,
                    duration_sec: 0.0,
                    stats,
                },
            );
            true
        }
        "error" => {
            let message = record
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("sidecar reported error")
                .to_string();
            let _ = app.emit(
                "landform://done",
                DoneEvent::Err {
                    classify_id: id.to_string(),
                    ok: false,
                    error: message,
                },
            );
            true
        }
        _ => false,
    }
}
