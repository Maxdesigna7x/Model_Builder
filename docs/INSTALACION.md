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
