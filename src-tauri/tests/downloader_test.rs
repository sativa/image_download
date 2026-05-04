use imagery_downloader_lib::core::downloader::{download_one, DownloadConfig};
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn config() -> DownloadConfig {
    DownloadConfig {
        max_retries: 3,
        backoff_base: Duration::from_millis(10),
        timeout_per_request: Duration::from_secs(2),
    }
}

#[tokio::test]
async fn download_one_succeeds_on_first_try() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1, 2, 3]))
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let bytes = download_one(&url, &config()).await.unwrap();
    assert_eq!(bytes.as_ref(), &[1, 2, 3]);
}

#[tokio::test]
async fn download_one_retries_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![9]))
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let bytes = download_one(&url, &config()).await.unwrap();
    assert_eq!(bytes.as_ref(), &[9]);
}

#[tokio::test]
async fn download_one_gives_up_after_max_retries() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/t"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let url = format!("{}/t", server.uri());
    let err = download_one(&url, &config()).await.unwrap_err();
    assert!(err.to_string().contains("503") || err.to_string().contains("retries"));
}

use imagery_downloader_lib::core::downloader::{download_all, ProgressUpdate};
use imagery_downloader_lib::core::sources::SourceKind;
use imagery_downloader_lib::core::tiles::TileCoord;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn download_all_empty_returns_empty() {
    let progress: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let p2 = progress.clone();
    let cfg = imagery_downloader_lib::core::downloader::DownloadConfig {
        max_retries: 0,
        backoff_base: std::time::Duration::ZERO,
        timeout_per_request: std::time::Duration::from_secs(1),
    };
    let result = download_all(
        vec![],
        SourceKind::Esri,
        cfg,
        4,
        CancellationToken::new(),
        move |p| p2.lock().unwrap().push(p),
    )
    .await;
    assert!(result.is_empty());
    assert!(progress.lock().unwrap().is_empty());
}

#[tokio::test]
async fn download_all_respects_cancellation() {
    let progress: Arc<Mutex<Vec<ProgressUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let p2 = progress.clone();
    let cancel = CancellationToken::new();
    cancel.cancel(); // pre-cancel

    let cfg = imagery_downloader_lib::core::downloader::DownloadConfig {
        max_retries: 0,
        backoff_base: std::time::Duration::ZERO,
        timeout_per_request: std::time::Duration::from_secs(1),
    };
    let result = download_all(
        (0..5).map(|x| TileCoord { x, y: 0, z: 5 }).collect(),
        SourceKind::Esri,
        cfg,
        4,
        cancel,
        move |p| p2.lock().unwrap().push(p),
    )
    .await;
    assert_eq!(result.len(), 5);
    assert!(result.iter().all(|t| t.bytes.is_none()));
    assert_eq!(progress.lock().unwrap().len(), 5);
}

use imagery_downloader_lib::core::downloader::TileCache;

#[tokio::test]
async fn tile_cache_missing_subset() {
    let cache = TileCache::new();
    let all = vec![
        TileCoord { x: 0, y: 0, z: 5 },
        TileCoord { x: 1, y: 0, z: 5 },
        TileCoord { x: 2, y: 0, z: 5 },
    ];
    cache.put(all[0], bytes::Bytes::from_static(&[1])).await;
    cache.put(all[2], bytes::Bytes::from_static(&[3])).await;
    let missing = cache.missing(&all).await;
    assert_eq!(missing, vec![all[1]]);
}
