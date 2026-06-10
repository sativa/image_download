#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Headless batch mode: `imagery-downloader batch --regions … --out …`.
    // Detected before any Tauri state is built so no webview/GUI window is
    // ever created. run_batch() owns its own Tokio runtime and exits the
    // process itself (diverging `-> !`).
    if imagery_downloader_lib::cli::is_batch_invocation() {
        imagery_downloader_lib::cli::run_batch();
    }
    // Headless classify mode: `imagery-downloader classify --input <tif|dir>` —
    // runs the trained model (python sidecar) over existing GeoTIFFs.
    if imagery_downloader_lib::cli::is_classify_invocation() {
        imagery_downloader_lib::cli::run_classify();
    }
    imagery_downloader_lib::run()
}
