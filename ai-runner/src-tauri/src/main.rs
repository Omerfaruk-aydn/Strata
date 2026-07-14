// AI Runner — Tauri Main
// Manages the Python backend process lifecycle.
// Phase 6: full sidecar integration.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::Manager;

/// Shared state for the backend process handle
struct BackendState {
    pid: Mutex<Option<u32>>,
}

#[tauri::command]
fn get_backend_status(state: tauri::State<BackendState>) -> String {
    let pid = state.pid.lock().unwrap();
    match *pid {
        Some(p) => format!("running:{}", p),
        None => "stopped".to_string(),
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(BackendState {
            pid: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_status])
        .setup(|app| {
            // In development mode, the Python backend runs separately.
            // In production, it will be launched as a sidecar here.
            #[cfg(not(debug_assertions))]
            {
                let sidecar_command = app.shell().sidecar("ai_runner_backend").unwrap();
                let (_rx, child) = sidecar_command
                    .args(["--host", "127.0.0.1", "--port", "8420"])
                    .spawn()
                    .expect("Failed to spawn backend sidecar");

                let state = app.state::<BackendState>();
                *state.pid.lock().unwrap() = Some(child.pid());

                println!("[Tauri] Backend sidecar started with PID {}", child.pid());
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
