// AI Runner — Tauri Main
// Manages the Python backend process lifecycle.
// Phase 6: full sidecar integration.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, RunEvent};

/// Shared state for the backend process handle
struct BackendState {
    pid: Mutex<Option<u32>>,
}

fn write_startup_error(message: &str) {
    let path = std::env::temp_dir().join("ai-runner-tauri-startup.log");
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{message}");
    }
}

#[cfg(not(debug_assertions))]
fn resolve_backend_executable<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<PathBuf, std::io::Error> {
    let current_executable = std::env::current_exe()?;
    let current_directory = current_executable
        .parent()
        .map(PathBuf::from)
        .unwrap_or_default();
    let resource_directory = app
        .path()
        .resource_dir()
        .map_err(|error| std::io::Error::other(error.to_string()))?;

    let filename = if cfg!(target_os = "windows") {
        "ai_runner_backend.exe"
    } else {
        "ai_runner_backend"
    };
    let candidates = [
        resource_directory.join(filename),
        current_directory.join(filename),
        current_directory.join("ai_runner_backend-x86_64-pc-windows-msvc.exe"),
    ];
    let candidate_description = format!("{candidates:?}");
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Backend sidecar bulunamadı. Aranan yollar: {candidate_description}"),
            )
        })
}

#[tauri::command]
fn get_backend_status(state: tauri::State<BackendState>) -> String {
    let pid = state.pid.lock().unwrap();
    match *pid {
        Some(p) => format!("running:{}", p),
        None => "stopped".to_string(),
    }
}

#[cfg(target_os = "windows")]
fn terminate_backend(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status();
}

#[cfg(not(target_os = "windows"))]
fn terminate_backend(pid: u32) {
    let _ = Command::new("kill").arg(pid.to_string()).status();
}

fn main() {
    write_startup_error("main:begin");
    let app = tauri::Builder::default()
        .manage(BackendState {
            pid: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![get_backend_status])
        .setup(|_app| {
            write_startup_error("setup:backend-hook");
            // In development mode, the Python backend runs separately.
            // In production, it will be launched as a sidecar here.
            #[cfg(not(debug_assertions))]
            {
                let desktop_executable = match std::env::current_exe() {
                    Ok(path) => path,
                    Err(error) => {
                        let message = format!("current_exe failed: {error}");
                        write_startup_error(&message);
                        return Err(std::io::Error::other(message).into());
                    }
                };
                let backend_executable = match resolve_backend_executable(_app.handle()) {
                    Ok(path) => path,
                    Err(error) => {
                        let message = format!("sidecar lookup failed: {error}");
                        write_startup_error(&message);
                        return Err(std::io::Error::other(message).into());
                    }
                };
                write_startup_error(&format!(
                    "sidecar path resolved: {}",
                    backend_executable.display()
                ));
                let child = match Command::new(&backend_executable)
                    .env("AI_RUNNER_DESKTOP_EXE", desktop_executable)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(child) => child,
                    Err(error) => {
                        let message = format!("sidecar spawn failed: {error}");
                        write_startup_error(&message);
                        return Err(std::io::Error::other(message).into());
                    }
                };

                let state = _app.state::<BackendState>();
                *state.pid.lock().unwrap() = Some(child.id());

                write_startup_error(&format!("sidecar started: pid={}", child.id()));
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::Exit = event {
            let state = app_handle.state::<BackendState>();
            let backend_pid = state.pid.lock().unwrap().take();
            if let Some(pid) = backend_pid {
                terminate_backend(pid);
            }
        }
    });
}
