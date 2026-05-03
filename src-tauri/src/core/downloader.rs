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

/// Fetch one URL with exponential backoff; up to `max_retries`. Returns bytes on success.
pub async fn download_one(url: &str, cfg: &DownloadConfig) -> Result<Bytes, DownloadError> {
    let client = reqwest::Client::builder()
        .timeout(cfg.timeout_per_request)
        .build()?;
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
