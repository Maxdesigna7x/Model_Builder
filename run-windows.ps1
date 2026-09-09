$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:MODELBUILDER_PYTHON = (Resolve-Path '.\.venv\Scripts\python.exe').Path
npm run tauri dev
