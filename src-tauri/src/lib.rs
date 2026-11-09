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

fn run_setup_command(_app: &AppHandle, command: &mut Command) -> Result<(), String> {
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
            // Detailed tool output is kept for errors; the UI receives structured progress.
        }
    }
    if !child.wait().map_err(|e| e.to_string())?.success() {
        return Err(format!("No se pudo completar la preparación. {last}"));
    }
    Ok(())
}

fn runtime_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("runtime"))
}

fn runtime_manifest(app: &AppHandle) -> Result<String, String> {
    let path = app
        .path()
        .resource_dir()
        .map_err(|e| e.to_string())?
        .join("pyproject.toml");
    std::fs::read_to_string(path)
        .map_err(|e| format!("No se encontró la configuración del motor: {e}"))
}

fn marker_backend(data: &[u8], manifest: &str) -> Option<String> {
    let value: Value = serde_json::from_slice(data).ok()?;
    let backend = value.get("backend")?.as_str()?;
    if value.get("manifest")?.as_str()? == manifest && ["cpu", "nvidia"].contains(&backend) {
        Some(backend.to_string())
    } else {
        None
    }
}

fn ready_backend(app: &AppHandle, manifest: &str) -> Option<String> {
    let data = std::fs::read(runtime_dir(app).ok()?.join("ready-manifest")).ok()?;
    if !private_python(app).ok()?.is_file() {
        return None;
    }
    marker_backend(&data, manifest)
}

#[cfg(test)]
mod setup_tests {
    use super::*;

    #[test]
    fn marker_remembers_the_selected_backend_and_rejects_old_or_stale_installations() {
        let marker =
            serde_json::to_vec(&json!({"manifest": "dependencies", "backend": "nvidia"})).unwrap();
        assert_eq!(
            marker_backend(&marker, "dependencies").as_deref(),
            Some("nvidia")
        );
        assert_eq!(marker_backend(&marker, "changed"), None);
        assert_eq!(marker_backend(b"old manifest", "old manifest"), None);
        let invalid =
            serde_json::to_vec(&json!({"manifest": "dependencies", "backend": "other"})).unwrap();
        assert_eq!(marker_backend(&invalid, "dependencies"), None);
    }
}

#[tauri::command]
async fn runtime_status(app: AppHandle) -> Result<Value, String> {
    if cfg!(debug_assertions) {
        return Ok(json!({"ready": true}));
    }
    let manifest = runtime_manifest(&app)?;
    let backend = ready_backend(&app, &manifest);
    Ok(json!({"ready": backend.is_some(), "backend": backend}))
}

fn setup_event(
    app: &AppHandle,
    stage: &str,
    message: &str,
    completed: usize,
    total: usize,
    current: &str,
) {
    let _ = app.emit("setup-progress", json!({"stage":stage,"message":message,"completed":completed,"total":total,"current":current}));
}

#[tauri::command]
async fn prepare_runtime(app: AppHandle, backend: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        if cfg!(debug_assertions) { return Ok(()); }
        if backend != "cpu" && backend != "nvidia" { return Err("Opción de dispositivo no válida".into()); }
        let _guard = SETUP_LOCK.lock().map_err(|e| e.to_string())?;
        let manifest = runtime_manifest(&app)?;
        if ready_backend(&app, &manifest).as_deref() == Some(backend.as_str()) { return Ok(()); }
        let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
        let runtime = runtime_dir(&app)?;
        let marker = runtime.join("ready-manifest");
        let python = private_python(&app)?;
        std::fs::create_dir_all(&runtime).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&marker);
        let uv = resources.join("runtime").join(if cfg!(windows) { "uv.exe" } else { "uv" });
        setup_event(&app, "python", "Preparando Python privado", 0, 0, "Python 3.12");
        let mut venv = Command::new(&uv);
        venv.arg("venv").arg("--clear").arg("--managed-python").arg("--python").arg("3.12").arg(runtime.join("venv"));
        venv.env("UV_PYTHON_INSTALL_DIR", runtime.join("python")).env("UV_NO_PROGRESS", "1").env("UV_HTTP_TIMEOUT", "300");
        run_setup_command(&app, &mut venv)?;
        let torch_backend = if backend == "nvidia" { "cu130" } else { "cpu" };
        let lockfile = runtime.join("resolved-requirements.txt");
        setup_event(&app, "resolve", "Calculando los componentes necesarios", 0, 0, "Dependencias de ModelBuilder");
        let mut resolve = Command::new(&uv);
        resolve.args(["pip", "compile", "--python"]).arg(&python).args(["--torch-backend", torch_backend, "--no-header", "--no-annotate", "--output-file"]).arg(&lockfile).arg(resources.join("pyproject.toml"));
        resolve.env("UV_NO_PROGRESS", "1").env("UV_HTTP_TIMEOUT", "300");
        run_setup_command(&app, &mut resolve)?;
        let lock = std::fs::read_to_string(&lockfile).map_err(|e| e.to_string())?;
        let packages: Vec<&str> = lock.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#')).collect();
        if packages.is_empty() || packages.iter().any(|p| !p.contains("==") || p.starts_with('-')) { return Err("No se pudo interpretar la lista de dependencias".into()); }
        let total = packages.len();
        for (index, package) in packages.iter().enumerate() {
            let name = package.split("==").next().unwrap_or(package);
            setup_event(&app, "install", "Descargando e instalando componentes", index, total, name);
            let mut install = Command::new(&uv);
            install.args(["pip", "install", "--no-deps", "--python"]).arg(&python).args(["--torch-backend", torch_backend]).arg(package);
            install.env("UV_NO_PROGRESS", "1").env("UV_HTTP_TIMEOUT", "300");
            run_setup_command(&app, &mut install).map_err(|e| format!("{name}: {e}"))?;
        }
        setup_event(&app, "verify", "Comprobando la instalación", total, total, "Motor de ModelBuilder");
        let mut dependencies = Command::new(&uv);
        dependencies.args(["pip", "check", "--python"]).arg(&python);
        run_setup_command(&app, &mut dependencies)?;
        let mut check = Command::new(&python);
        let verify = if backend == "nvidia" {
            "import torch, torchvision, numpy, PIL, datasets, transformers, ultralytics, cv2; assert torch.version.cuda and torch.cuda.is_available(), 'No se detectó una GPU NVIDIA compatible. Comprueba el controlador o selecciona CPU.'; torch.ones(1, device='cuda').cpu()"
        } else {
            "import torch, torchvision, numpy, PIL, datasets, transformers, ultralytics, cv2"
        };
        check.args(["-c", verify]);
        run_setup_command(&app, &mut check)?;
        let ready = json!({"manifest":manifest, "backend":backend});
        std::fs::write(marker, serde_json::to_vec(&ready).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        Ok(())
    }).await.map_err(|e| e.to_string())?
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
            runtime_status,
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
