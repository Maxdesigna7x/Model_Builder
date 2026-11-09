use serde_json::{json, Value};
use std::sync::Mutex;
use tauri::AppHandle;
use tauri_plugin_updater::UpdaterExt;

// A single gate prevents training and update installation from starting together.
pub static ACTIVITY: Mutex<usize> = Mutex::new(0);

#[tauri::command]
pub async fn app_update_check(app: AppHandle) -> Result<Value, String> {
    if cfg!(debug_assertions) {
        return Ok(Value::Null);
    }
    let update = app
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| e.to_string())?;
    Ok(match update {
        Some(update) => {
            json!({"version":update.version,"notes":update.body,"canInstall": cfg!(windows) || std::env::var_os("APPIMAGE").is_some()})
        }
        None => Value::Null,
    })
}

#[tauri::command]
pub async fn app_update_install(app: AppHandle) -> Result<(), String> {
    if cfg!(debug_assertions) {
        return Err("Actualización desactivada en modo desarrollo".into());
    }
    if cfg!(target_os = "linux") && std::env::var_os("APPIMAGE").is_none() {
        return Err("Actualiza el paquete DEB desde Releases o tu gestor de paquetes.".into());
    }
    {
        let mut activity = ACTIVITY.lock().map_err(|e| e.to_string())?;
        if *activity != 0 {
            return Err("Termina o detén el entrenamiento antes de actualizar.".into());
        }
        *activity = usize::MAX;
    }
    let result = async {
        let update = app
            .updater()
            .map_err(|e| e.to_string())?
            .check()
            .await
            .map_err(|e| e.to_string())?
            .ok_or("No hay una actualización disponible")?;
        update
            .download_and_install(|_, _| {}, || {})
            .await
            .map_err(|e| e.to_string())?;
        Ok::<(), String>(())
    }
    .await;
    if result.is_err() {
        if let Ok(mut activity) = ACTIVITY.lock() {
            *activity = 0;
        }
    }
    result?;
    app.restart();
}
