pub mod commands;
pub mod core;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                // Silence reqwest's per-response TRACE noise (e.g. 'shouldn't retry!'
                // fires for every successful response — useless for users).
                .level_for("reqwest", log::LevelFilter::Info)
                .level_for("reqwest::retry", log::LevelFilter::Warn)
                .level_for("reqwest::connect", log::LevelFilter::Info)
                .level_for("hyper", log::LevelFilter::Info)
                .level_for("hyper_util", log::LevelFilter::Info)
                .level_for("rustls", log::LevelFilter::Info)
                .level_for("h2", log::LevelFilter::Info)
                // tao's window-event spam is also too verbose by default.
                .level_for("tao", log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(commands::runner::Runner::default())
        .invoke_handler(tauri::generate_handler![
            commands::download::estimate_output,
            commands::download::start_download,
            commands::download::cancel_download,
            commands::download::retry_failed,
            commands::sources::probe_sources,
            commands::vector::parse_vector_file,
            commands::history::list_history,
            commands::history::clear_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
