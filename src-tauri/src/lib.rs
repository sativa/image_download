pub mod commands;
pub mod core;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(commands::runner::Runner::default())
        .invoke_handler(tauri::generate_handler![
            commands::download::estimate_output,
            commands::download::start_download,
            commands::download::cancel_download,
            commands::download::retry_failed,
            commands::vector::parse_vector_file,
            commands::history::list_history,
            commands::history::clear_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
