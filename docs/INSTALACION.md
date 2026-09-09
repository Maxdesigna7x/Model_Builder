# Instalación

## Requisitos comunes

- Git.
- Node.js 20 o superior (npm incluido).
- Python 3.11 o superior, con venv y pip.
- Rust estable y Cargo, necesarios para compilar Tauri.
- Internet durante la instalación y la primera descarga de datasets.

## Linux (Ubuntu/Debian)

    chmod +x scripts/install-linux.sh run.sh run-web.sh
    ./scripts/install-linux.sh
    ./run.sh

El instalador configura WebKitGTK y las librerías nativas de Tauri, crea .venv, instala el backend con pip install -e . y ejecuta npm ci.

## Windows 10/11

También se requiere WebView2 Runtime, Visual Studio Build Tools 2022 con “Desktop development with C++” y el SDK de Windows. Instala Rust desde https://rustup.rs, Node.js 20 LTS y Python 3.11+ agregándolos al PATH.

En PowerShell:

    Set-ExecutionPolicy -Scope Process Bypass
    .\scripts\install-windows.ps1
    .\run-windows.ps1

Los instaladores son repetibles y usan package-lock.json mediante npm ci.
