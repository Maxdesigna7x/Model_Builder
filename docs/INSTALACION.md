# Instalación

## Requisitos comunes

- Git.
- Node.js 20.19+ en la rama 20, o 22.12+ (se recomienda LTS 22+) (npm incluido).
- Python 3.11 o superior, con venv y pip.
- Rust estable y Cargo, necesarios para compilar Tauri.
- Internet durante la instalación y la primera descarga de datasets.

## Linux (Ubuntu/Debian)

    chmod +x scripts/install-linux.sh run.sh run-web.sh
    ./scripts/install-linux.sh
    ./run.sh

El instalador configura WebKitGTK y las librerías nativas de Tauri, crea .venv, instala el backend con pip install -e . y ejecuta npm ci.

## Windows 10/11

También se requiere WebView2 Runtime, Visual Studio Build Tools 2022 con “Desktop development with C++” y el SDK de Windows. Instala Rust desde https://rustup.rs, Node.js 22 LTS y Python 3.11+ agregándolos al PATH.

En PowerShell:

    Set-ExecutionPolicy -Scope Process Bypass
    .\scripts\install-windows.ps1
    .\run-windows.ps1

Los instaladores son repetibles y usan package-lock.json mediante npm ci.

## Compilar la aplicación de escritorio

Para preparar un instalador, instala Node, Rust y uv 0.9.30 en el equipo de compilación. En Linux instala además los paquetes nativos indicados arriba. Ejecuta:

    npm ci
    python -m pip install uv==0.9.30
    python scripts/prepare-desktop-runtime.py
    npm run tauri -- build

En Linux puede usarse `python3` en vez de `python`. Los paquetes generados aparecen en `src-tauri/target/release/bundle/`. Compila Windows en Windows y Linux en Linux. El flujo manual **Desktop builds** de GitHub Actions genera artefactos para ambos sistemas y publica automáticamente una nueva versión en GitHub Releases cuando ambas compilaciones terminan correctamente. Se inicia con `gh workflow run desktop.yml --ref main`. Cada ejecución usa una etiqueta propia y una versión de instalador basada en su número de ejecución.

La versión distribuida prepara un Python privado al primer inicio y descarga las dependencias; necesita conexión y varios GB libres. La instalación ofrece elegir PyTorch para CPU o para una GPU NVIDIA compatible. Los proyectos se guardan en la carpeta de datos de usuario de ModelBuilder, separados de la aplicación. El modo web y el entorno de desarrollo siguen usando los comandos anteriores. Las aplicaciones aún requieren pruebas en equipos limpios antes de su distribución pública.

Al primer inicio de la aplicación distribuida se elige **CPU** o **GPU NVIDIA**. La opción NVIDIA instala la variante CUDA 13.0 de PyTorch y comprueba que una GPU y un controlador compatibles estén disponibles. Si la comprobación falla, se puede reintentar o elegir CPU. Esta descarga necesita conexión y puede ocupar varios GB. Los usuarios de la versión anterior verán la selección al actualizar; los proyectos permanecen en su carpeta de datos.

Durante la instalación, el porcentaje cuenta los paquetes preparados sobre el total resuelto, incluidos los que ya estaban en caché. El indicador del paquete actual es indeterminado porque no se recibe un porcentaje fiable de su descarga. Para verificar CUDA en un equipo con NVIDIA, puede ejecutarse `python scripts/test-desktop-runtime.py --backend nvidia`; esto crea un entorno temporal y descarga las dependencias necesarias.


## Diagnóstico de arranque en Linux

Si la ventana queda en blanco, cierra la aplicación y prueba desde una terminal:

    modelbuilder --graphics=compatible

Para AppImage, usa la ruta al archivo en lugar de `modelbuilder`. Este modo cambia el renderizado de la interfaz; no desactiva CUDA para entrenamiento. Para ver datos básicos del entorno sin descargar dependencias:

    modelbuilder --diagnose

El modo compatible es una primera prueba; no garantiza resolver todos los problemas gráficos o de bibliotecas.

## Actualizaciones y conservación del entorno

La aplicación comprueba si hay una versión nueva al entrar en la interfaz principal. También se puede usar **Actualizaciones**. Windows y AppImage permiten descargar, verificar la firma e instalar la actualización con reinicio. Para DEB se abre la página de Releases y se instala el paquete nuevo con el gestor del sistema. Se requiere una actualización manual inicial para versiones antiguas que todavía no incluyen este mecanismo.

Guarda los cambios y termina tus tareas antes de actualizar. La instalación desde la app se bloquea mientras hay entrenamiento activo. Si no hay conexión, la app sigue funcionando y se puede intentar comprobar más tarde.

Python, las bibliotecas y los proyectos permanecen en la carpeta de datos de usuario. Los cambios en el nombre, descripción o versión de la app no invalidan el entorno. Cuando cambian los requisitos de dependencias, se prepara otro entorno y se activa solo después de comprobarlo; el anterior se conserva. Se reutilizan Python y los paquetes disponibles en la caché, aunque un cambio real de dependencias puede necesitar descargas. No borres los datos de usuario durante una actualización.

Este mecanismo conserva los archivos de proyectos; los cambios futuros de sus formatos o de checkpoints necesitarán migraciones y pruebas específicas. No garantiza compatibilidad de cualquier checkpoint con cualquier versión futura de PyTorch.

El workflow genera paquetes de actualización firmados y `latest.json`. La clave privada se guarda en el secreto `TAURI_SIGNING_PRIVATE_KEY` de GitHub y la pública en `tauri.conf.json`. La firma del actualizador es independiente de la firma de editor de Windows. No subas la clave privada al repositorio ni la regeneres en cada compilación. Existe una copia privada local en `~/.codex/modelbuilder-updater.key`; debe respaldarse de forma segura para conservar la continuidad de las actualizaciones.
