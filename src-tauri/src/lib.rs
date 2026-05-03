mod history;
mod mocks;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(mocks::runner::Runner::default())
        .invoke_handler(tauri::generate_handler![
            mocks::commands::estimate_output,
            mocks::commands::start_download,
            mocks::commands::cancel_download,
            mocks::commands::retry_failed,
            mocks::commands::parse_vector_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
