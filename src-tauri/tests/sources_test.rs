use imagery_downloader_lib::core::sources::*;
use std::time::Duration;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn probe_returns_latency_under_response_delay() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/probe"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(50)))
        .mount(&server)
        .await;

    let url = format!("{}/probe", server.uri());
    let lat = probe_url(&url).await.expect("probe ok");
    assert!(lat >= Duration::from_millis(40), "got {:?}", lat);
    assert!(lat < Duration::from_secs(2));
}

#[tokio::test]
async fn probe_returns_err_on_404() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/probe"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let url = format!("{}/probe", server.uri());
    assert!(probe_url(&url).await.is_err());
}
