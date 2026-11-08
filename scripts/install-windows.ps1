$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
function Require-Command($Name, $Label) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) { throw "$Label no está instalado o no está en PATH." }
}
Require-Command node 'Node.js'
$NodeVersion = node -p "process.versions.node"
& node -e 'const [m,n]=process.versions.node.split(".").map(Number);process.exit((m===20&&n>=19)||(m>=22&&(m>22||n>=12))?0:1)'
if ($LASTEXITCODE -ne 0) { throw "Node.js 20.19+ (rama 20) o 22.12+ requerido; encontrado $NodeVersion." }
Require-Command npm 'npm'
$Py = Get-Command py -ErrorAction SilentlyContinue
if ($Py) { $PyArgs = @() } else { $Py = Get-Command python -ErrorAction SilentlyContinue; $PyArgs = @() }
if (-not $Py) { throw 'Python 3.11+ no está instalado o no está en PATH.' }
$PyVersion = & $Py.Source @PyArgs -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'
if ([version]$PyVersion -lt [version]'3.11.0') { throw "Python 3.11+ requerido; encontrado $PyVersion." }
Require-Command cargo 'Rust/Cargo'
Require-Command rustc 'rustc'
Write-Host '==> Entorno Python'
& $Py.Source @PyArgs -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .
Write-Host '==> Dependencias JavaScript'
npm ci
Write-Host 'Instalación completada. Ejecuta .\run-windows.ps1 o npm run dev.'
