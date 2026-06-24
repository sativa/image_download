//! Headless batch-download CLI.
//!
//! Invoked as:
//!   imagery-downloader batch --regions <regions.json> --out <dir> \
//!       [--source esri|google|auto] [--zoom 17] [--concurrency N]
//!
//! Drives the *exact same* core pipeline the Tauri GUI uses
//! (`core::{tiles, downloader, stitcher, cog}`) so output rasters are
//! byte-for-byte equivalent to a GUI download: stitched z-tiles cropped to
//! the user bbox, written as an EPSG:3857 GeoTIFF. No webview is created.
//!
//! Regions JSON accepts either:
//!   * a top-level array of cells, or
//!   * an object with `train` and/or `test` arrays.
//! Each cell needs at least `{"county": str, "idx": int, "bbox": [w,s,e,n]}`
//! (bbox in WGS84 lon/lat). Output per cell:
//!   <out>/{county}_{idx}[_{source}].tif
//! Files that already exist are skipped (resumable across runs).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use crate::core::cog::{bbox_3857_from_range, crop_to_user_bbox, write_cog, Compression, CogParams};
use crate::core::downloader::{download_all, DownloadConfig};
use crate::core::sources::{pick_auto, SourceKind};
use crate::core::stitcher::stitch_rgba;
use crate::core::tiles::range_for_bbox;

/// One cell to download. `idx` accepts a JSON number or numeric string.
#[derive(Debug, Clone, Deserialize)]
struct Region {
    county: String,
    #[serde(deserialize_with = "de_idx")]
    idx: i64,
    bbox: [f64; 4],
}

/// `idx` is normally an integer but some upstream JSON emits it as a string;
/// accept either so we don't choke on a mixed list.
fn de_idx<'de, D>(d: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| D::Error::custom("idx not an integer")),
        serde_json::Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map_err(|e| D::Error::custom(format!("idx string not an integer: {e}"))),
        _ => Err(D::Error::custom("idx must be a number or numeric string")),
    }
}

/// Either a bare array of regions or `{train:[...], test:[...]}`. Unknown
/// extra keys on the object form are ignored.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RegionsFile {
    Array(Vec<Region>),
    Split {
        #[serde(default)]
        train: Vec<Region>,
        #[serde(default)]
        test: Vec<Region>,
    },
}

impl RegionsFile {
    fn into_regions(self) -> Vec<Region> {
        match self {
            RegionsFile::Array(v) => v,
            RegionsFile::Split { mut train, test } => {
                train.extend(test);
                train
            }
        }
    }
}

struct BatchArgs {
    regions: PathBuf,
    out: PathBuf,
    /// "esri" | "google" | "auto"
    source: String,
    zoom: u32,
    concurrency: usize,
    /// Per-tile network retries (matches GUI default of 3).
    retry_per_tile: u32,
    /// "jpeg" | "deflate" | "none" — output GeoTIFF compression.
    compress: String,
    /// JPEG quality 1..=100 (only used when compress=jpeg).
    quality: u8,
    /// `--classify [backend]`: after downloading, run the trained model over every cell tif
    /// (same python sidecar the GUI uses) and emit per-parcel polygons. None = download only.
    classify: Option<String>,
    /// Post-download quality control: detect blank/black/missing cells and re-fetch them
    /// (cross-source fallback) so the mosaic has no blank blocks. Default on.
    qc: bool,
    /// A cell whose mean RGB brightness (0..255) is below this is treated as blank/black
    /// and re-fetched. Default 20 (real imagery is ~60-120; bad tiles ~0-19).
    qc_brightness: f32,
    /// During QC, if a cell is still blank after re-fetching the primary source, try the
    /// other source (esri↔google) into the same file. Default on.
    qc_fallback: bool,
}

impl BatchArgs {
    /// Resolve the validated `--compress`/`--quality` flags into a [`Compression`].
    fn compression(&self) -> Compression {
        match self.compress.as_str() {
            "deflate" => Compression::Deflate,
            "none" => Compression::None,
            _ => Compression::Jpeg {
                quality: self.quality,
            },
        }
    }
}

/// True when argv looks like a batch invocation (`<prog> batch ...`).
/// Kept tiny so `main.rs` can branch before building any Tauri state.
pub fn is_batch_invocation() -> bool {
    std::env::args().nth(1).as_deref() == Some("batch")
}

const USAGE: &str = "\
imagery-downloader batch — headless bulk tile download

USAGE:
    imagery-downloader batch --regions <regions.json> --out <dir> \\
        [--source esri|google|auto] [--zoom <8-23>] [--concurrency <N>] \\
        [--retry-per-tile <N>] [--compress jpeg|deflate|none] [--quality <1-100>] \\
        [--no-qc] [--qc-brightness <0-255>] [--no-qc-fallback]

OPTIONS:
    --regions <PATH>     JSON: array of cells, or {train:[...],test:[...]}.
                         Each cell: {\"county\":str,\"idx\":int,\"bbox\":[w,s,e,n]}
    --out <DIR>          Output directory (created if missing).
    --source <S>         esri | google | auto   [default: esri]
    --zoom <Z>           XYZ zoom level 8..23    [default: 17  (~1 m/px)]
    --concurrency <N>    Parallel tile fetches per cell [default: 16]
    --retry-per-tile <N> Network retries per tile  [default: 3]
    --compress <C>       jpeg (lossy YCbCr, 3-band, ~10x) | deflate (lossless,
                         3-band, ~2x) | none (uncompressed RGBA 4-band)
                         [default: jpeg]
    --quality <Q>        JPEG quality 1..100 (compress=jpeg only) [default: 95]
    --no-qc              Disable post-download QC (blank/black/missing cell
                         detection + re-fetch). QC is ON by default.
    --qc-brightness <B>  Cells with mean RGB brightness < B (0..255) are treated
                         as blank/black and re-fetched [default: 20].
    --no-qc-fallback     During QC, don't try the other source (esri<->google)
                         for cells still blank after re-fetching the primary.
    --classify [B]       After downloading, classify with the trained model.
                         Trained backends (parcel_dist|cropland|parcel_bh|parcel|
                         landcover) now produce ONE SEAMLESS COUNTY coverage:
                         all cells are merged into a single mosaic, run through a
                         global-watershed inference + topology-preserving vectorise
                         + curve smoothing (Chaikin) + standard postproc (sliver/
                         gap-hole/invalid) -> <out>/county_seamless.parquet. This
                         replaces the old per-cell (cell-seamed) output.
                         B = parcel_dist (default, best: dist-peak watershed) |
                         cropland | parcel_bh | parcel | landcover.
                         (Pipeline env: IMG_PIPELINE_PYTHON; needs geopandas+topojson.)
    -h, --help           Show this help.

Output: one EPSG:3857 GeoTIFF per cell, named
    {county}_{idx}[_{source}].tif   (skips files that already exist)

See also: imagery-downloader classify --help   (classify existing GeoTIFFs)";

/// Parse argv after the leading `batch` token. Returns `Err(message)` for
/// bad/missing flags; the caller prints it and exits non-zero.
fn parse_args(argv: &[String]) -> Result<BatchArgs, String> {
    let mut regions: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut source = "esri".to_string();
    let mut zoom: u32 = 17;
    let mut concurrency: usize = 16;
    let mut retry_per_tile: u32 = 3;
    let mut compress = "jpeg".to_string();
    let mut quality: u8 = 95;
    let mut classify: Option<String> = None;
    let mut qc = true;
    let mut qc_brightness: f32 = 20.0;
    let mut qc_fallback = true;

    // Grab the value following the flag at index `i`, advancing `i` past it.
    fn value_at(argv: &[String], i: &mut usize, name: &str) -> Result<String, String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} requires a value"))
    }

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(USAGE.to_string()),
            "--regions" => regions = Some(PathBuf::from(value_at(argv, &mut i, "--regions")?)),
            "--out" | "--out-dir" => out = Some(PathBuf::from(value_at(argv, &mut i, "--out")?)),
            "--source" => source = value_at(argv, &mut i, "--source")?,
            "--zoom" => {
                zoom = value_at(argv, &mut i, "--zoom")?
                    .parse()
                    .map_err(|_| "--zoom must be an integer".to_string())?
            }
            "--concurrency" => {
                concurrency = value_at(argv, &mut i, "--concurrency")?
                    .parse()
                    .map_err(|_| "--concurrency must be an integer".to_string())?
            }
            "--retry-per-tile" => {
                retry_per_tile = value_at(argv, &mut i, "--retry-per-tile")?
                    .parse()
                    .map_err(|_| "--retry-per-tile must be an integer".to_string())?
            }
            "--compress" => compress = value_at(argv, &mut i, "--compress")?,
            "--classify" => {
                // Optional value: bare `--classify` defaults to the best backend (parcel_dist).
                let next = argv.get(i + 1).map(String::as_str);
                classify = Some(match next {
                    Some(v) if !v.starts_with("--") => {
                        i += 1;
                        v.to_string()
                    }
                    _ => "parcel_dist".to_string(),
                });
            }
            "--quality" => {
                quality = value_at(argv, &mut i, "--quality")?
                    .parse()
                    .map_err(|_| "--quality must be an integer".to_string())?
            }
            "--no-qc" => qc = false,
            "--no-qc-fallback" => qc_fallback = false,
            "--qc-brightness" => {
                qc_brightness = value_at(argv, &mut i, "--qc-brightness")?
                    .parse()
                    .map_err(|_| "--qc-brightness must be a number".to_string())?
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
        i += 1;
    }

    let regions = regions.ok_or_else(|| format!("--regions is required\n\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("--out is required\n\n{USAGE}"))?;
    if !(8..=23).contains(&zoom) {
        return Err(format!("--zoom {zoom} out of range 8..=23"));
    }
    if concurrency == 0 {
        return Err("--concurrency must be >= 1".to_string());
    }
    match source.as_str() {
        "esri" | "google" | "auto" => {}
        other => return Err(format!("--source must be esri|google|auto, got {other}")),
    }
    match compress.as_str() {
        "jpeg" | "deflate" | "none" => {}
        other => return Err(format!("--compress must be jpeg|deflate|none, got {other}")),
    }
    if !(1..=100).contains(&quality) {
        return Err(format!("--quality {quality} out of range 1..=100"));
    }
    if !(0.0..=255.0).contains(&qc_brightness) {
        return Err(format!("--qc-brightness {qc_brightness} out of range 0..=255"));
    }
    if let Some(b) = &classify {
        match b.as_str() {
            "parcel_dist" | "cropland" | "parcel_bh" | "parcel" | "landcover" => {}
            other => return Err(format!("--classify backend must be parcel_dist|cropland|parcel_bh|parcel|landcover, got {other}")),
        }
    }
    Ok(BatchArgs {
        regions,
        out,
        source,
        zoom,
        concurrency,
        retry_per_tile,
        compress,
        quality,
        classify,
        qc,
        qc_brightness,
        qc_fallback,
    })
}

/// Headless batch entry. Builds its own multi-thread Tokio runtime (the GUI's
/// runtime is owned by Tauri, which we never start here) and processes every
/// cell sequentially — each cell already fans its tiles out across
/// `--concurrency` connections, which saturates the network without
/// overlapping whole-cell stitches in memory.
///
/// Exit code: 0 if every requested cell ends up on disk (downloaded, cached,
/// or skipped), non-zero if any cell failed to produce a raster.
pub fn run_batch() -> ! {
    let argv: Vec<String> = std::env::args().skip(2).collect(); // drop prog + "batch"
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            // --help prints usage and exits 0; real errors exit 2.
            let code = if msg == USAGE { 0 } else { 2 };
            std::process::exit(code);
        }
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("failed to start async runtime: {e}");
            std::process::exit(1);
        }
    };

    let code = rt.block_on(async move { run_batch_async(args).await });
    std::process::exit(code);
}

/// Outcome of one cell, for the run summary.
enum CellResult {
    /// Wrote a tif. `brightness` is the mean RGB (0..255) of the stitched image —
    /// QC uses it to flag blank/black cells.
    Wrote { brightness: f32 },
    Skipped,
    Failed,
}

/// Mean RGB brightness (0..255) over an image. A blank/black tile (source served
/// no imagery, or every tile request failed → transparent) scores near 0; real
/// imagery is ~60-120. Transparent pixels count as 0 so half-missing cells score low too.
fn mean_brightness(img: &image::RgbaImage) -> f32 {
    let mut sum: u64 = 0;
    for px in img.pixels() {
        let [r, g, b, a] = px.0;
        // Treat transparent (failed-tile) pixels as black.
        if a == 0 {
            continue;
        }
        sum += (r as u64 + g as u64 + b as u64) / 3;
    }
    let n = (img.width() as u64) * (img.height() as u64);
    if n == 0 {
        0.0
    } else {
        sum as f32 / n as f32
    }
}

async fn run_batch_async(args: BatchArgs) -> i32 {
    // Resolve source up-front: "auto" probes both once for the whole batch.
    let source: SourceKind = match args.source.as_str() {
        "auto" => {
            eprintln!("[batch] probing sources (auto)…");
            pick_auto().await
        }
        s => SourceKind::parse(s).expect("validated in parse_args"),
    };

    // Read + parse the regions file.
    let raw = match std::fs::read_to_string(&args.regions) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[batch] cannot read {}: {e}", args.regions.display());
            return 1;
        }
    };
    let regions = match serde_json::from_str::<RegionsFile>(&raw) {
        Ok(rf) => rf.into_regions(),
        Err(e) => {
            eprintln!("[batch] invalid regions JSON {}: {e}", args.regions.display());
            return 1;
        }
    };
    if regions.is_empty() {
        eprintln!("[batch] regions file has 0 cells — nothing to do");
        return 0;
    }

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("[batch] cannot create out dir {}: {e}", args.out.display());
        return 1;
    }

    let compression = args.compression();
    eprintln!(
        "[batch] {} cells | source={} | zoom={} | concurrency={} | compress={}{} | out={}",
        regions.len(),
        source.as_str(),
        args.zoom,
        args.concurrency,
        args.compress,
        if args.compress == "jpeg" {
            format!(" q{}", args.quality)
        } else {
            String::new()
        },
        args.out.display(),
    );

    let started = Instant::now();
    let mut wrote = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    // Cells that need a QC re-fetch: the download failed (no tif) or the stitched
    // image is blank/black (brightness below threshold). Stored as region indices.
    let mut bad: Vec<usize> = Vec::new();

    for (n, region) in regions.iter().enumerate() {
        let out_path = cell_output_path(&args.out, region, source);
        let label = format!(
            "{}/{} {}",
            n + 1,
            regions.len(),
            out_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
        );
        match download_one_cell(
            region,
            source,
            args.zoom,
            args.concurrency,
            args.retry_per_tile,
            compression,
            &out_path,
            false,
        )
        .await
        {
            CellResult::Wrote { brightness } => {
                if args.qc && brightness < args.qc_brightness {
                    bad.push(n);
                    eprintln!("[batch] {label}  ⚠ blank (brightness {brightness:.0} < {:.0}) — queued for QC", args.qc_brightness);
                } else {
                    wrote += 1;
                    eprintln!("[batch] {label}  ✓ wrote");
                }
            }
            CellResult::Skipped => {
                skipped += 1;
                eprintln!("[batch] {label}  • skip (exists)");
            }
            CellResult::Failed => {
                if args.qc {
                    bad.push(n);
                    eprintln!("[batch] {label}  ⚠ failed — queued for QC");
                } else {
                    failed += 1;
                    eprintln!("[batch] {label}  ✗ FAILED");
                }
            }
        }
    }

    // ── QC pass: re-fetch every blank/failed cell (primary source, then the other
    //    source as a fallback) so the mosaic has no blank/black blocks. ──
    if args.qc && !bad.is_empty() {
        eprintln!(
            "[batch][qc] {} blank/failed cell(s); re-fetching (threshold {:.0}, fallback {})…",
            bad.len(),
            args.qc_brightness,
            if args.qc_fallback { source.alternate().as_str() } else { "off" },
        );
        for &n in &bad {
            let region = &regions[n];
            let out_path = cell_output_path(&args.out, region, source);
            let label = out_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let mut best = -1.0f32; // brightness of the best attempt so far
            // 1) re-fetch the primary source a couple of times (transient blanks/gaps).
            for _ in 0..2 {
                if let CellResult::Wrote { brightness } = download_one_cell(
                    region, source, args.zoom, args.concurrency, args.retry_per_tile,
                    compression, &out_path, true,
                ).await {
                    best = best.max(brightness);
                    if brightness >= args.qc_brightness {
                        break;
                    }
                }
            }
            // 2) still blank → try the other provider into the same file.
            if best < args.qc_brightness && args.qc_fallback {
                let alt = source.alternate();
                if let CellResult::Wrote { brightness } = download_one_cell(
                    region, alt, args.zoom, args.concurrency, args.retry_per_tile,
                    compression, &out_path, true,
                ).await {
                    if brightness >= args.qc_brightness {
                        best = brightness;
                        eprintln!("[batch][qc] {label}  ✓ fixed via {} (brightness {brightness:.0})", alt.as_str());
                    } else {
                        best = best.max(brightness);
                    }
                }
            }
            if best >= args.qc_brightness {
                wrote += 1;
                eprintln!("[batch][qc] {label}  ✓ ok (brightness {best:.0})");
            } else {
                failed += 1;
                eprintln!("[batch][qc] {label}  ✗ STILL BLANK (best brightness {best:.0}) — no imagery in either source?");
            }
        }
    } else if !bad.is_empty() {
        // QC disabled: count the queued-bad cells as failures.
        failed += bad.len();
    }

    eprintln!(
        "[batch] done: {wrote} wrote, {skipped} skipped, {failed} failed in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    // Chained classification. Trained (dense) backends now emit ONE seamless county coverage via
    // parcel_pipeline.py (global watershed + smooth + postproc); single-image backends (slic/sam3/
    // dino) stay per-cell. The county pipeline merges all downloaded cells into one mosaic — so it
    // is *not* seamed at cell borders, unlike the old per-cell path.
    let mut classify_failed = 0usize;
    if let Some(backend) = &args.classify {
        let opts = SidecarOpts::from_env();
        if is_county_backend(backend) {
            if !run_county_pipeline(&args.out, backend, &opts) {
                classify_failed = 1;
            }
        } else {
            classify_failed = classify_dir(&args.out, backend, &opts);
        }
    }
    if failed > 0 || classify_failed > 0 {
        1
    } else {
        0
    }
}

/// Build the on-disk path for one cell.
///
/// `{county}_{idx}_{source}.tif`. The `_{source}` suffix mirrors the Python
/// reference (`dl_hires_cells.py`) so multi-source batches don't collide and
/// the filename records which provider produced the imagery.
fn cell_output_path(out_dir: &Path, region: &Region, source: SourceKind) -> PathBuf {
    let safe_county: String = region
        .county
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    out_dir.join(format!("{}_{}_{}.tif", safe_county, region.idx, source.as_str()))
}

/// Download + stitch + georeference one cell, writing an EPSG:3857 GeoTIFF.
/// Reuses the GUI core verbatim. Returns Skipped if the file already exists.
async fn download_one_cell(
    region: &Region,
    source: SourceKind,
    zoom: u32,
    concurrency: usize,
    retry_per_tile: u32,
    compression: Compression,
    out_path: &Path,
    force: bool,
) -> CellResult {
    if !force && out_path.exists() {
        return CellResult::Skipped;
    }
    let bbox = region.bbox;
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] {
        eprintln!(
            "[batch]   invalid bbox for {}_{}: {:?} (need [w,s,e,n] with w<e, s<n)",
            region.county, region.idx, bbox
        );
        return CellResult::Failed;
    }

    let range = range_for_bbox(bbox, zoom);
    let coords: Vec<_> = range.iter().collect();
    let total = coords.len();

    let cfg = DownloadConfig {
        max_retries: retry_per_tile,
        backoff_base: Duration::from_millis(200),
        timeout_per_request: Duration::from_secs(30),
    };

    // No cancellation in batch mode; pass an unused token.
    let token = CancellationToken::new();
    let tiles = download_all(coords, source, cfg, concurrency.max(1), token, |_p| {}).await;

    let failed_tiles = tiles.iter().filter(|t| t.bytes.is_none()).count();
    if failed_tiles == total {
        eprintln!(
            "[batch]   all {total} tiles failed for {}_{} (network / no imagery?)",
            region.county, region.idx
        );
        return CellResult::Failed;
    }
    if failed_tiles > 0 {
        eprintln!(
            "[batch]   warning: {failed_tiles}/{total} tiles failed for {}_{} (transparent gaps)",
            region.county, region.idx
        );
    }

    // Stitch on whole-tile boundaries, then crop to the user's exact bbox —
    // identical to the GUI's run_pipeline so georef matches byte-for-byte.
    let img = stitch_rgba(&tiles, range);
    let outer_3857 = bbox_3857_from_range(range);
    let (img, bbox_3857) = crop_to_user_bbox(img, outer_3857, bbox);

    let brightness = mean_brightness(&img);
    match write_cog(&img, &CogParams { bbox_3857, zoom }, compression, out_path) {
        Ok(()) => CellResult::Wrote { brightness },
        Err(e) => {
            eprintln!(
                "[batch]   write_cog failed for {}: {e}",
                out_path.display()
            );
            CellResult::Failed
        }
    }
}

// ───────────────────────── classification (trained model → polygons) ─────────────────────────
//
// `imagery-downloader classify …` and `batch --classify` both spawn the *same* python sidecar the
// GUI uses (`python -m sam3_classify`), so CLI products are identical to GUI products: per-parcel
// polygons as GeoParquet + GPKG + GeoJSON + class raster + legend. Sequential per tif (the sidecar
// saturates the GPU/CPU by itself); resumable (skips tifs whose .landform.parquet already exists).

/// Paths the sidecar needs. Hard-coded dev defaults, each overridable by env var
/// (same vars as the GUI: SAM3_PYTHON / SAM3_SIDECAR_DIR / CROPLAND_BACKBONE / *_WEIGHTS).
struct SidecarOpts {
    python: String,
    /// Python env that runs `parcel_pipeline.py` (the seamless county pipeline).
    /// Needs geopandas + topojson + shapely 2.1 + rasterio + torch; the SAM3 env lacks
    /// topojson, so this defaults to a fuller env (override with IMG_PIPELINE_PYTHON).
    pipeline_python: String,
    sidecar_dir: PathBuf,
    backbone: String,
    device: String,
}

impl SidecarOpts {
    fn from_env() -> Self {
        Self {
            python: std::env::var("SAM3_PYTHON")
                .unwrap_or_else(|_| "/Users/zhangfeng/D/sam3/sam3_env_py312/bin/python".into()),
            pipeline_python: std::env::var("IMG_PIPELINE_PYTHON")
                .unwrap_or_else(|_| "/Users/zhangfeng/miniconda3/envs/py312/bin/python".into()),
            sidecar_dir: std::env::var("SAM3_SIDECAR_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from("/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar")
                }),
            backbone: std::env::var("CROPLAND_BACKBONE")
                .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/dinov3-vitl16-sat493m".into()),
            device: "auto".into(),
        }
    }
}

/// Backends whose product is a *county-wide seamless* polygon coverage (trained dense models).
/// These now run the global-watershed + Chaikin-smooth + postproc pipeline over the whole cell
/// directory at once (one merged mosaic), instead of the old per-cell (seamed) classification.
/// Single-image backends (slic/sam3/dino) are NOT in this set — they stay per-cell.
fn is_county_backend(backend: &str) -> bool {
    matches!(
        backend,
        "parcel_dist" | "landcover" | "cropland" | "parcel_bh" | "parcel"
    )
}

/// Default checkpoint per backend (mirrors commands::classify so GUI and CLI agree).
fn weights_for(backend: &str) -> String {
    match backend {
        "parcel_dist" => std::env::var("PARCEL_DIST_WEIGHTS")
            .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/parcel_dist.pt".into()),
        "landcover" | "parcel_bh" => std::env::var("LANDCOVER_WEIGHTS")
            .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/landcover8_bh.pt".into()),
        "cropland" | "parcel" => std::env::var("CROPLAND_WEIGHTS")
            .unwrap_or_else(|_| "/Users/zhangfeng/D/cropland_dino/cropland_gdlxff.pt".into()),
        _ => std::env::var("SAM3_WEIGHTS")
            .unwrap_or_else(|_| "/Users/zhangfeng/D/sam3/sam3_weights/sam3.pt".into()),
    }
}

/// Classify ONE GeoTIFF via the sidecar. Streams its stage/done/error NDJSON lines through
/// (suppresses the very chatty per-tile progress records). Returns true on success.
fn classify_one(input: &Path, output: &Path, backend: &str, opts: &SidecarOpts) -> bool {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let mut cmd = Command::new(&opts.python);
    cmd.current_dir(&opts.sidecar_dir)
        .arg("-m")
        .arg("sam3_classify")
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--weights")
        .arg(weights_for(backend))
        .arg("--backbone-dir")
        .arg(&opts.backbone)
        .arg("--backend")
        .arg(backend)
        .arg("--device")
        .arg(&opts.device)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[classify]   ✗ cannot spawn sidecar ({}): {e}", opts.python);
            return false;
        }
    };
    let mut ok = false;
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            if line.contains("\"type\":\"done\"") {
                ok = true;
                println!("{line}");
            } else if line.contains("\"type\":\"stage\"") || line.contains("\"type\":\"error\"") {
                println!("{line}");
            }
        }
    }
    let status_ok = child.wait().map(|s| s.success()).unwrap_or(false);
    ok && status_ok
}

/// Classify every plain GeoTIFF in `dir` (skips previous products). Returns #failed.
fn classify_dir(dir: &Path, backend: &str, opts: &SidecarOpts) -> usize {
    let mut tifs: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().is_some_and(|x| x == "tif")
                        && !p.file_name().is_some_and(|n| {
                            let n = n.to_string_lossy();
                            n.contains(".landform") || n.contains("_mosaic")
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    tifs.sort();
    if tifs.is_empty() {
        eprintln!("[classify] no input tifs in {}", dir.display());
        return 0;
    }
    eprintln!("[classify] {} tifs | backend={backend} | weights={}", tifs.len(), weights_for(backend));
    let started = Instant::now();
    let (mut done, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    for (i, tif) in tifs.iter().enumerate() {
        let stem = tif.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let output = dir.join(format!("{stem}.landform.tif"));
        let parquet = dir.join(format!("{stem}.landform.parquet"));
        if parquet.exists() {
            skipped += 1;
            continue; // resumable
        }
        eprintln!("[classify] {}/{} {stem}", i + 1, tifs.len());
        if classify_one(tif, &output, backend, opts) {
            done += 1;
        } else {
            failed += 1;
            eprintln!("[classify]   ✗ FAILED {stem}");
        }
    }
    eprintln!(
        "[classify] done: {done} classified, {skipped} skipped, {failed} failed in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    failed
}

/// Run the **seamless county pipeline** (`parcel_pipeline.py`) over a whole cell directory:
/// rasterio-merge all cell tifs into one mosaic, global-watershed inference, topology-preserving
/// vectorise + Chaikin smooth + standard postproc (sliver/gap/invalid) -> ONE seamless county
/// GeoParquet. Replaces the old per-cell (seamed) classification for trained backends.
///
/// Output: `<dir>/county_seamless.parquet`. Returns true on success.
fn run_county_pipeline(dir: &Path, backend: &str, opts: &SidecarOpts) -> bool {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let out_parquet = dir.join("county_seamless.parquet");
    if out_parquet.exists() {
        eprintln!(
            "[county] {} already exists — skip (delete to re-run)",
            out_parquet.display()
        );
        return true;
    }
    eprintln!(
        "[county] seamless pipeline | backend={backend} | cells-dir={} | python={}",
        dir.display(),
        opts.pipeline_python
    );
    let mut cmd = Command::new(&opts.pipeline_python);
    cmd.current_dir(&opts.sidecar_dir)
        .arg("parcel_pipeline.py")
        .arg("--cells-dir")
        .arg(dir)
        .arg("--weights")
        .arg(weights_for(backend))
        .arg("--backbone")
        .arg(&opts.backbone)
        .arg("--out")
        .arg(&out_parquet)
        .arg("--device")
        .arg(&opts.device)
        .arg("--boundary")
        .arg("none")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[county]   ✗ cannot spawn pipeline ({}): {e}",
                opts.pipeline_python
            );
            return false;
        }
    };
    // Stream the pipeline's stage prints straight through.
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            println!("{line}");
        }
    }
    let ok = child.wait().map(|s| s.success()).unwrap_or(false);
    if ok {
        eprintln!("[county]   ✓ wrote {}", out_parquet.display());
    } else {
        eprintln!("[county]   ✗ pipeline FAILED");
    }
    ok
}

const CLASSIFY_USAGE: &str = "\
imagery-downloader classify — run the trained land-cover model on GeoTIFFs

USAGE:
    imagery-downloader classify --input <tif|dir> [--backend parcel_dist] \\
        [--out <dir>] [--device auto|cpu|mps|cuda]

OPTIONS:
    --input <PATH>   One GeoTIFF, or a DIRECTORY of cells. For a directory + a
                     trained backend (parcel_dist|cropland|parcel_bh|parcel|
                     landcover) the cells are merged into one mosaic and run
                     through the SEAMLESS county pipeline (global watershed +
                     smoothing + postproc) -> <dir>/county_seamless.parquet.
                     Single-image backends (slic/sam3/dino) classify each tif.
    --backend <B>    parcel_dist (default — dist-peak watershed) | cropland |
                     parcel_bh | parcel | landcover.
    --out <DIR>      Output dir [default: alongside each input].
    --device <D>     auto (default) | cpu | mps | cuda.
    -h, --help       Show this help.

Env overrides: SAM3_PYTHON, SAM3_SIDECAR_DIR, CROPLAND_BACKBONE,
PARCEL_DIST_WEIGHTS / LANDCOVER_WEIGHTS / CROPLAND_WEIGHTS / SAM3_WEIGHTS.

Output per input: <stem>.landform.{parquet,gpkg,geojson,tif,legend.json}
(GeoParquet is the primary vector product).";

/// True when argv looks like `<prog> classify ...`.
pub fn is_classify_invocation() -> bool {
    std::env::args().nth(1).as_deref() == Some("classify")
}

/// Headless classify entry (`imagery-downloader classify …`). Exits the process.
pub fn run_classify() -> ! {
    let argv: Vec<String> = std::env::args().skip(2).collect();
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut backend = "parcel_dist".to_string();
    let mut opts = SidecarOpts::from_env();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                eprintln!("{CLASSIFY_USAGE}");
                std::process::exit(0);
            }
            "--input" => {
                i += 1;
                input = argv.get(i).map(PathBuf::from);
            }
            "--out" | "--out-dir" => {
                i += 1;
                out = argv.get(i).map(PathBuf::from);
            }
            "--backend" => {
                i += 1;
                backend = argv.get(i).cloned().unwrap_or_default();
            }
            "--device" => {
                i += 1;
                opts.device = argv.get(i).cloned().unwrap_or_else(|| "auto".into());
            }
            other => {
                eprintln!("unknown argument: {other}\n\n{CLASSIFY_USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let Some(input) = input else {
        eprintln!("--input is required\n\n{CLASSIFY_USAGE}");
        std::process::exit(2);
    };
    let failed = if input.is_dir() {
        // A directory of cells -> trained backends produce one seamless county coverage;
        // single-image backends (slic/sam3/dino) still classify each tif per-cell.
        if is_county_backend(&backend) {
            usize::from(!run_county_pipeline(&input, &backend, &opts))
        } else {
            classify_dir(&input, &backend, &opts)
        }
    } else {
        let dir = out.unwrap_or_else(|| input.parent().unwrap_or(Path::new(".")).to_path_buf());
        let _ = std::fs::create_dir_all(&dir);
        let stem = input.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let output = dir.join(format!("{stem}.landform.tif"));
        usize::from(!classify_one(&input, &output, &backend, &opts))
    };
    std::process::exit(if failed > 0 { 1 } else { 0 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_array_regions() {
        let j = r#"[{"county":"620105","idx":0,"bbox":[103.751,36.118,103.771,36.138]}]"#;
        let rf: RegionsFile = serde_json::from_str(j).unwrap();
        let regs = rf.into_regions();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].county, "620105");
        assert_eq!(regs[0].idx, 0);
        assert_eq!(regs[0].bbox, [103.751, 36.118, 103.771, 36.138]);
    }

    #[test]
    fn parses_train_test_split() {
        let j = r#"{"train":[{"county":"A","idx":1,"bbox":[0.0,0.0,1.0,1.0]}],
                    "test":[{"county":"B","idx":2,"bbox":[2.0,2.0,3.0,3.0]}]}"#;
        let regs = serde_json::from_str::<RegionsFile>(j).unwrap().into_regions();
        assert_eq!(regs.len(), 2);
        assert_eq!(regs[0].county, "A");
        assert_eq!(regs[1].county, "B");
    }

    #[test]
    fn idx_accepts_string() {
        let j = r#"[{"county":"X","idx":"7","bbox":[0.0,0.0,1.0,1.0]}]"#;
        let regs = serde_json::from_str::<RegionsFile>(j).unwrap().into_regions();
        assert_eq!(regs[0].idx, 7);
    }

    #[test]
    fn output_name_includes_source() {
        let r = Region {
            county: "620105".into(),
            idx: 3,
            bbox: [0.0, 0.0, 1.0, 1.0],
        };
        let p = cell_output_path(Path::new("/tmp/o"), &r, SourceKind::Esri);
        assert_eq!(p, Path::new("/tmp/o/620105_3_esri.tif"));
        let p2 = cell_output_path(Path::new("/tmp/o"), &r, SourceKind::Google);
        assert_eq!(p2, Path::new("/tmp/o/620105_3_google.tif"));
    }

    #[test]
    fn unsafe_county_chars_sanitised() {
        let r = Region {
            county: "a/b c".into(),
            idx: 0,
            bbox: [0.0, 0.0, 1.0, 1.0],
        };
        let p = cell_output_path(Path::new("/tmp"), &r, SourceKind::Esri);
        assert_eq!(p.file_name().unwrap(), "a_b_c_0_esri.tif");
    }

    #[test]
    fn qc_on_by_default() {
        let a = parse_args(&["--regions".into(), "r.json".into(), "--out".into(), "/tmp".into()]).unwrap();
        assert!(a.qc);
        assert_eq!(a.qc_brightness, 20.0);
        assert!(a.qc_fallback);
    }

    #[test]
    fn qc_flags_parse() {
        let a = parse_args(&[
            "--regions".into(), "r.json".into(), "--out".into(), "/tmp".into(),
            "--no-qc".into(), "--qc-brightness".into(), "35".into(), "--no-qc-fallback".into(),
        ]).unwrap();
        assert!(!a.qc);
        assert_eq!(a.qc_brightness, 35.0);
        assert!(!a.qc_fallback);
    }

    #[test]
    fn qc_brightness_range_validated() {
        assert!(parse_args(&[
            "--regions".into(), "r.json".into(), "--out".into(), "/tmp".into(),
            "--qc-brightness".into(), "999".into(),
        ]).is_err());
    }

    #[test]
    fn mean_brightness_flags_blank() {
        use image::{Rgba, RgbaImage};
        let black = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 255]));
        let bright = RgbaImage::from_pixel(8, 8, Rgba([90, 90, 90, 255]));
        let transparent = RgbaImage::from_pixel(8, 8, Rgba([0, 0, 0, 0]));
        assert!(mean_brightness(&black) < 20.0);
        assert!(mean_brightness(&bright) >= 20.0);
        assert!(mean_brightness(&transparent) < 20.0); // failed tiles -> blank
    }

    #[test]
    fn source_alternate_flips() {
        assert_eq!(SourceKind::Esri.alternate(), SourceKind::Google);
        assert_eq!(SourceKind::Google.alternate(), SourceKind::Esri);
    }

    #[test]
    fn parse_args_requires_regions_and_out() {
        assert!(parse_args(&["--out".into(), "/tmp".into()]).is_err());
        assert!(parse_args(&["--regions".into(), "r.json".into()]).is_err());
        let ok = parse_args(&[
            "--regions".into(),
            "r.json".into(),
            "--out".into(),
            "/tmp".into(),
        ])
        .unwrap();
        assert_eq!(ok.zoom, 17);
        assert_eq!(ok.source, "esri");
        assert_eq!(ok.concurrency, 16);
    }

    #[test]
    fn parse_args_rejects_bad_zoom_and_source() {
        assert!(parse_args(&[
            "--regions".into(),
            "r".into(),
            "--out".into(),
            "/tmp".into(),
            "--zoom".into(),
            "99".into(),
        ])
        .is_err());
        assert!(parse_args(&[
            "--regions".into(),
            "r".into(),
            "--out".into(),
            "/tmp".into(),
            "--source".into(),
            "bing".into(),
        ])
        .is_err());
    }
}
