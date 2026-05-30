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
        [--retry-per-tile <N>] [--compress jpeg|deflate|none] [--quality <1-100>]

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
    -h, --help           Show this help.

Output: one EPSG:3857 GeoTIFF per cell, named
    {county}_{idx}[_{source}].tif   (skips files that already exist)";

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
            "--quality" => {
                quality = value_at(argv, &mut i, "--quality")?
                    .parse()
                    .map_err(|_| "--quality must be an integer".to_string())?
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
    Ok(BatchArgs {
        regions,
        out,
        source,
        zoom,
        concurrency,
        retry_per_tile,
        compress,
        quality,
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
    Wrote,
    Skipped,
    Failed,
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
        )
        .await
        {
            CellResult::Wrote => {
                wrote += 1;
                eprintln!("[batch] {label}  ✓ wrote");
            }
            CellResult::Skipped => {
                skipped += 1;
                eprintln!("[batch] {label}  • skip (exists)");
            }
            CellResult::Failed => {
                failed += 1;
                eprintln!("[batch] {label}  ✗ FAILED");
            }
        }
    }

    eprintln!(
        "[batch] done: {wrote} wrote, {skipped} skipped, {failed} failed in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    if failed > 0 {
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
) -> CellResult {
    if out_path.exists() {
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

    match write_cog(&img, &CogParams { bbox_3857, zoom }, compression, out_path) {
        Ok(()) => CellResult::Wrote,
        Err(e) => {
            eprintln!(
                "[batch]   write_cog failed for {}: {e}",
                out_path.display()
            );
            CellResult::Failed
        }
    }
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
