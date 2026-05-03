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
