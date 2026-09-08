use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Emitter, Manager};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn python_command() -> String {
    std::env::var("MODELBUILDER_PYTHON").unwrap_or_else(|_| "python".into())
}

fn runtime_paths(app: &AppHandle) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let bundled_engine = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("backend/modelbuilder/engine.py");
    let bundled = bundled_engine.is_file();
    let engine = if bundled {
        bundled_engine
    } else {
        project_root().join("backend/modelbuilder/engine.py")
    };
    let workdir = engine
        .parent()
        .and_then(|path| path.parent())
        .ok_or("No se pudo resolver el directorio del backend")?
        .to_path_buf();
    let workspace = if bundled {
        app.path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("workspace_data")
    } else {
        project_root().join("workspace_data")
    };
    std::fs::create_dir_all(&workspace).map_err(|e| e.to_string())?;
    Ok((engine, workdir, workspace))
}

fn execute(app: &AppHandle, request: Value) -> Result<Value, String> {
    let (engine, workdir, workspace) = runtime_paths(app)?;
    let mut command = Command::new(python_command());
    command.arg(engine).current_dir(workdir);
    if std::env::var_os("MODELBUILDER_WORKSPACE").is_none() {
        command.env("MODELBUILDER_WORKSPACE", workspace);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("No se pudo iniciar Python: {e}"))?;
    {
        let mut stdin = child.stdin.take().ok_or("Python no abrió stdin")?;
        stdin
            .write_all(request.to_string().as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().last().ok_or_else(|| {
        format!(
            "Python no respondió: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    serde_json::from_str(line).map_err(|e| format!("Respuesta inválida del backend: {e}"))
}

#[tauri::command]
async fn backend_request(app: AppHandle, request: Value) -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(move || execute(&app, request))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn pick_directory() -> Result<Option<String>, String> {
    let folder = tauri::async_runtime::spawn_blocking(|| rfd::FileDialog::new().pick_folder())
        .await
        .map_err(|e| e.to_string())?;
    Ok(folder.map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
async fn start_training(app: AppHandle, request: Value) -> Result<Value, String> {
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            let (engine, workdir, workspace) = runtime_paths(&app)?;
            let mut command = Command::new(python_command());
            command.arg(engine).current_dir(workdir);
            if std::env::var_os("MODELBUILDER_WORKSPACE").is_none() {
                command.env("MODELBUILDER_WORKSPACE", workspace);
            }
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| e.to_string())?;
            {
                let mut stdin = child.stdin.take().ok_or("Python no abrió stdin")?;
                stdin
                    .write_all(request.to_string().as_bytes())
                    .map_err(|e| e.to_string())?;
            }
            let stdout = child.stdout.take().ok_or("Python no abrió stdout")?;
            for line in BufReader::new(stdout).lines() {
                let line = line.map_err(|e| e.to_string())?;
                match serde_json::from_str::<Value>(&line) {
                    Ok(event) => {
                        let _ = app.emit("training-event", event);
                    }
                    Err(_) => {
                        let _ = app.emit("training-event", json!({"type":"log","message":line}));
                    }
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            if !status.success() {
                return Err(format!("El entrenamiento terminó con {status}"));
            }
            Ok(())
        })();
        if let Err(message) = result {
            let _ = app.emit("training-event", json!({"type":"error","message":message}));
        }
    });
    Ok(json!({"started": true}))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            backend_request,
            start_training,
            pick_directory
        ])
        .run(tauri::generate_context!())
        .expect("error al ejecutar ModelBuilder");
}
