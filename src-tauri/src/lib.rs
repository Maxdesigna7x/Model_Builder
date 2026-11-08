use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;

static SETUP_LOCK: Mutex<()> = Mutex::new(());
use tauri::{AppHandle, Emitter, Manager};

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn private_python(app: &AppHandle) -> Result<PathBuf, String> {
    let env = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("runtime")
        .join("venv");
    Ok(if cfg!(windows) {
        env.join("Scripts/python.exe")
    } else {
        env.join("bin/python")
    })
}

fn python_command(app: &AppHandle) -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        return Ok(std::env::var_os("MODELBUILDER_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python")));
    }
    private_python(app)
}

fn hide_console(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    #[cfg(not(windows))]
    let _ = command;
}

fn run_setup_command(app: &AppHandle, command: &mut Command) -> Result<(), String> {
    hide_console(command);
    command.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|e| format!("No se pudo iniciar la preparación: {e}"))?;
    let stderr = child.stderr.take().ok_or("No se pudo leer el progreso")?;
    let mut last = String::new();
    for line in BufReader::new(stderr).lines() {
        let line = line.map_err(|e| e.to_string())?;
        if !line.trim().is_empty() {
            last = line.clone();
            let _ = app.emit("setup-progress", line);
        }
    }
    if !child.wait().map_err(|e| e.to_string())?.success() {
        return Err(format!("No se pudo completar la preparación. {last}"));
    }
    Ok(())
}

#[tauri::command]
async fn prepare_runtime(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        if cfg!(debug_assertions) {
            return Ok(());
        }
        let _guard = SETUP_LOCK.lock().map_err(|e| e.to_string())?;
        let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
        let manifest = resources.join("pyproject.toml");
        let requirements = std::fs::read(&manifest)
            .map_err(|e| format!("No se encontró la configuración del motor: {e}"))?;
        let runtime = app
            .path()
            .app_data_dir()
            .map_err(|e| e.to_string())?
            .join("runtime");
        let marker = runtime.join("ready-manifest");
        let python = private_python(&app)?;
        if python.is_file()
            && std::fs::read(&marker).ok().as_deref() == Some(requirements.as_slice())
        {
            return Ok(());
        }
        std::fs::create_dir_all(&runtime).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&marker);
        let uv = resources
            .join("runtime")
            .join(if cfg!(windows) { "uv.exe" } else { "uv" });
        let _ = app.emit("setup-progress", "Preparando Python privado…");
        let mut venv = Command::new(&uv);
        venv.arg("venv")
            .arg("--clear")
            .arg("--managed-python")
            .arg("--python")
            .arg("3.12")
            .arg(runtime.join("venv"));
        venv.env("UV_PYTHON_INSTALL_DIR", runtime.join("python"))
            .env("UV_NO_PROGRESS", "1");
        run_setup_command(&app, &mut venv)?;
        let _ = app.emit(
            "setup-progress",
            "Descargando e instalando dependencias. Esto puede tardar varios minutos…",
        );
        let mut install = Command::new(&uv);
        install
            .arg("pip")
            .arg("install")
            .arg("--python")
            .arg(&python)
            .arg("--torch-backend")
            .arg("cpu")
            .arg("-r")
            .arg(&manifest);
        install
            .env("UV_NO_PROGRESS", "1")
            .env("UV_HTTP_TIMEOUT", "300");
        run_setup_command(&app, &mut install)?;
        let mut check = Command::new(&python);
        check.args([
            "-c",
            "import torch, torchvision, numpy, PIL, datasets, transformers, ultralytics, cv2",
        ]);
        run_setup_command(&app, &mut check)?;
        std::fs::write(marker, requirements).map_err(|e| e.to_string())?;
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
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
    let mut command = Command::new(python_command(app)?);
    hide_console(&mut command);
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
async fn pick_dataset_file() -> Result<Option<String>, String> {
    let file = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .add_filter("Datos", &["csv", "tsv", "txt"])
            .pick_file()
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(file.map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
async fn pick_video_file() -> Result<Option<String>, String> {
    let file = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .add_filter(
                "Video",
                &["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg", "m4v"],
            )
            .pick_file()
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(file.map(|path| path.to_string_lossy().to_string()))
}

#[tauri::command]
async fn start_training(app: AppHandle, request: Value) -> Result<Value, String> {
    std::thread::spawn(move || {
        let result = (|| -> Result<(), String> {
            let (engine, workdir, workspace) = runtime_paths(&app)?;
            let mut command = Command::new(python_command(&app)?);
            hide_console(&mut command);
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
            prepare_runtime,
            backend_request,
            start_training,
            pick_directory,
            pick_dataset_file,
            pick_video_file
        ])
        .run(tauri::generate_context!())
        .expect("error al ejecutar ModelBuilder");
}
