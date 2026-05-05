//! Parallel tile downloader with retry, cancellation and tile cache.

use bytes::Bytes;
use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub max_retries: u32,
    pub backoff_base: Duration,
    pub timeout_per_request: Duration,
}

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("exhausted {0} retries; last status {1}")]
    Exhausted(u32, u16),
    #[error("cancelled")]
    Cancelled,
}

/// Build a reqwest client suitable for tile fetching: keep-alive enabled,
/// reasonable per-request timeout, real-looking UA so Google / ESRI don't
/// 403 us, and HTTP/2 prior knowledge disabled (some CDNs misbehave on h2).
pub fn build_client(cfg: &DownloadConfig) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .timeout(cfg.timeout_per_request)
        .user_agent(concat!(
            "imagery-downloader/",
            env!("CARGO_PKG_VERSION"),
            " (+https://github.com/sativa/image_download)"
        ))
        .pool_max_idle_per_host(16)
        .build()
}

/// Fetch one URL with exponential backoff; up to `max_retries`. Returns bytes on success.
pub async fn download_one(
    client: &reqwest::Client,
    url: &str,
    cfg: &DownloadConfig,
) -> Result<Bytes, DownloadError> {
    let mut last_status: u16 = 0;
    for attempt in 0..=cfg.max_retries {
        let resp = client.get(url).send().await;
        match resp {
            Ok(r) if r.status().is_success() => return Ok(r.bytes().await?),
            Ok(r) => last_status = r.status().as_u16(),
            Err(_e) if attempt < cfg.max_retries => {}
            Err(e) => return Err(DownloadError::Network(e)),
        }
        if attempt < cfg.max_retries {
            sleep(cfg.backoff_base * 2u32.pow(attempt)).await;
        }
    }
    Err(DownloadError::Exhausted(cfg.max_retries, last_status))
}

use crate::core::sources::{url_for_tile, SourceKind};
use crate::core::tiles::TileCoord;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct DownloadedTile {
    pub coord: TileCoord,
    pub bytes: Option<Bytes>, // None if all retries failed
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub completed: u32,
    pub total: u32,
    pub bytes_downloaded: u64,
    pub last_failed: Option<TileCoord>,
}

pub async fn download_all<F>(
    coords: Vec<TileCoord>,
    source: SourceKind,
    cfg: DownloadConfig,
    max_concurrency: usize,
    cancel: CancellationToken,
    on_progress: F,
) -> Vec<DownloadedTile>
where
    F: FnMut(ProgressUpdate) + Send + 'static,
{
    download_all_with_sink(
        coords,
        source,
        cfg,
        max_concurrency,
        cancel,
        on_progress,
        |_, _| {}, // no-op tile sink: caller doesn't care about per-tile bytes
    )
    .await
}

/// Same as `download_all`, plus a callback invoked exactly once per
/// successfully downloaded tile, *before* the tile is forwarded to the
/// progress aggregator. Use this to persist tile bytes to disk for resume.
///
/// The sink runs on the channel-receiver task (single-threaded) so
/// implementations don't need to be Sync / handle reentrancy. It runs
/// synchronously — keep it short. For larger work (e.g. fsync), spawn_blocking.
pub async fn download_all_with_sink<F, S>(
    coords: Vec<TileCoord>,
    source: SourceKind,
    cfg: DownloadConfig,
    max_concurrency: usize,
    cancel: CancellationToken,
    mut on_progress: F,
    mut on_tile_success: S,
) -> Vec<DownloadedTile>
where
    F: FnMut(ProgressUpdate) + Send + 'static,
    S: FnMut(TileCoord, &Bytes) + Send + 'static,
{
    let total = coords.len() as u32;
    let cfg = Arc::new(cfg);
    let completed = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let bytes_total = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (tx, mut rx) = tokio::sync::mpsc::channel::<DownloadedTile>(max_concurrency * 2);

    // Single shared client across all tiles — keep-alive + connection pooling.
    // Without this, every tile spawned a new TCP+TLS handshake and Google/ESRI
    // would 429 / drop us within seconds.
    let client = match build_client(&cfg) {
        Ok(c) => Arc::new(c),
        Err(_) => {
            // Build failed — return all tiles as failed without progress emits.
            return coords
                .into_iter()
                .map(|c| DownloadedTile {
                    coord: c,
                    bytes: None,
                })
                .collect();
        }
    };

    let cancel_inner = cancel.clone();
    let cfg_inner = cfg.clone();
    let client_inner = client.clone();
    let driver = tokio::spawn(async move {
        stream::iter(coords)
            .for_each_concurrent(max_concurrency, |c| {
                let tx = tx.clone();
                let cfg = cfg_inner.clone();
                let cancel = cancel_inner.clone();
                let client = client_inner.clone();
                async move {
                    if cancel.is_cancelled() {
                        let _ = tx
                            .send(DownloadedTile {
                                coord: c,
                                bytes: None,
                            })
                            .await;
                        return;
                    }
                    let url = url_for_tile(source, c);
                    let bytes = tokio::select! {
                        r = download_one(&client, &url, &cfg) => r.ok(),
                        _ = cancel.cancelled() => None,
                    };
                    let _ = tx.send(DownloadedTile { coord: c, bytes }).await;
                }
            })
            .await;
    });

    let mut out = Vec::with_capacity(total as usize);
    while let Some(tile) = rx.recv().await {
        let nb = tile.bytes.as_ref().map(|b| b.len() as u64).unwrap_or(0);
        // Persist successful tile *before* counting it as completed — that way
        // a crash between the WAL append and the progress emit just causes a
        // duplicate progress event, never a phantom-completed tile.
        if let Some(b) = tile.bytes.as_ref() {
            on_tile_success(tile.coord, b);
        }
        let new_bytes = bytes_total.fetch_add(nb, std::sync::atomic::Ordering::Relaxed) + nb;
        let new_completed = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let last_failed = if tile.bytes.is_none() {
            Some(tile.coord)
        } else {
            None
        };
        on_progress(ProgressUpdate {
            completed: new_completed,
            total,
            bytes_downloaded: new_bytes,
            last_failed,
        });
        out.push(tile);
    }
    let _ = driver.await;
    out
}

use std::collections::HashMap;
use tokio::sync::Mutex as TokioMutex;

/// Session-scoped cache. Successful downloads stay; retry_failed only fetches missing.
pub struct TileCache {
    inner: TokioMutex<HashMap<TileCoord, Bytes>>,
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TileCache {
    pub fn new() -> Self {
        Self {
            inner: TokioMutex::new(HashMap::new()),
        }
    }
    pub async fn put(&self, c: TileCoord, b: Bytes) {
        self.inner.lock().await.insert(c, b);
    }
    pub async fn get(&self, c: TileCoord) -> Option<Bytes> {
        self.inner.lock().await.get(&c).cloned()
    }
    pub async fn missing(&self, all: &[TileCoord]) -> Vec<TileCoord> {
        let g = self.inner.lock().await;
        all.iter().filter(|c| !g.contains_key(c)).copied().collect()
    }
}
