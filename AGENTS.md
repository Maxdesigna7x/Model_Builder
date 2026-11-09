# Instrucciones para agentes

## Aplicaciones de escritorio e instaladores

- ModelBuilder mantiene la interfaz web local y la aplicación de escritorio Tauri. Conserva ambas al introducir cambios.
- El código fuente de la interfaz está en `frontend/`, el motor Python en `backend/modelbuilder/` y la configuración de escritorio en `src-tauri/`. Los proyectos, datasets y checkpoints son datos locales; no los incluyas en los instaladores ni en Git.
- Para generar instaladores de Windows y Linux desde el mismo cambio, usa el flujo `.github/workflows/desktop.yml` de GitHub Actions. Primero asegúrate de que el commit que se debe compilar esté en la rama remota solicitada; GitHub Actions no puede compilar cambios que solo existen en el equipo local. Inicia el flujo con:

  ```bash
  gh workflow run desktop.yml --ref main
  ```

  Sustituye `main` por la rama o referencia remota que corresponda. El flujo también se activa al subir etiquetas `desktop-v*`. Genera `.deb` y `.AppImage` en Linux, y `.exe` y `.msi` en Windows; sube los artefactos de la ejecución y publica automáticamente una nueva versión en GitHub Releases cuando ambas compilaciones terminan correctamente. Cada ejecución usa una etiqueta propia y actualiza la versión del instalador con el número de ejecución; no reemplaza los archivos de releases anteriores.
- **No esperes a que termine una compilación de instaladores**, ni sigas consultando su estado, salvo que el usuario solicite explícitamente esperar o verificar el resultado. Inicia el flujo y comunica que quedó solicitado, junto con el enlace a Actions si está disponible. No afirmes que los instaladores se generaron o pasaron las pruebas hasta comprobarlo.
- Para compilar localmente, utiliza Windows para los instaladores Windows y Linux para los instaladores Linux. Instala los requisitos de desarrollo de Tauri, Node/npm, Python y Rust; luego ejecuta desde la raíz:

  ```bash
  npm ci
  python -m pip install uv==0.9.30
  python scripts/prepare-desktop-runtime.py
  npm run tauri -- build
  ```

  En Linux, utiliza `python3` si `python` no está disponible. Los archivos se generan bajo `src-tauri/target/release/bundle/`, en `deb/`, `appimage/`, `nsis/` o `msi/` según el sistema. Si se pide iniciar una compilación local sin esperar, lánzala en segundo plano y guarda su salida en un archivo de registro.
- Estos pasos son repetibles, pero no garantizan compilaciones idénticas byte por byte ni dependencias Python idénticas: hay versiones Python expresadas como rangos y se resuelven al preparar el entorno. No describas el proceso como reproducible de forma estricta sin fijar y verificar todas las dependencias.
- La aplicación distribuida prepara su propio Python al primer inicio y permite elegir CPU o NVIDIA CUDA. Mantén sus comprobaciones y la información de progreso al modificar el instalador. Las descargas de CUDA pueden ser grandes; no realices una instalación completa de prueba salvo que sea necesaria para el cambio o que el usuario la solicite.
- El flujo y los comandos de instalación se describen también en `docs/INSTALACION.md`. Si cambian, actualiza esa guía y este archivo.

- Para problemas de ventana en blanco en Linux, probar `modelbuilder --graphics=compatible` y obtener información básica con `modelbuilder --diagnose`. Estos comandos son una mejora inicial; no declarar resuelto un caso sin verificarlo en el equipo afectado. El trabajo pendiente está en `docs/22-plan-arranque-portable-linux.md`.
- Las actualizaciones usan `tauri-plugin-updater`, artefactos `.sig` y `latest.json`, generados por el workflow. No regenerar la clave: la pública está en `tauri.conf.json` y la privada en el secreto `TAURI_SIGNING_PRIVATE_KEY`. Nunca incluir la clave privada en Git ni en registros. Los clientes anteriores al actualizador requieren una instalación manual inicial.
- El entorno se valida por el contrato de dependencias, no por la versión de la app. Los entornos nuevos se preparan en carpetas separadas y se activan tras verificarlos; conservar los anteriores y todos los proyectos. No añadir migraciones de datos destructivas al actualizador.
