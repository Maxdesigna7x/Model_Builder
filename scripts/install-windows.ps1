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
if ($Py) {
  $PyArgs = @('-3.12')
  $PyVersion = & $Py.Source @PyArgs -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>$null
  if ($LASTEXITCODE -ne 0) {
    $PyArgs = @('-3.11')
    $PyVersion = & $Py.Source @PyArgs -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>$null
  }
  if ($LASTEXITCODE -ne 0) {
    $PyArgs = @()
    $PyVersion = & $Py.Source @PyArgs -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>$null
  }
} else {
  $Py = Get-Command python -ErrorAction SilentlyContinue
  $PyArgs = @()
  if ($Py) { $PyVersion = & $Py.Source -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' }
}
if (-not $Py -or $LASTEXITCODE -ne 0) { throw 'Python 3.11+ no está instalado o no está en PATH.' }
if ([version]$PyVersion -lt [version]'3.11.0') { throw "Python 3.11+ requerido; encontrado $PyVersion." }
Require-Command cargo 'Rust/Cargo'
Require-Command rustc 'rustc'
Write-Host '==> Entorno Python'
& $Py.Source @PyArgs -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear .venv.' }
$TorchIndex = 'https://download.pytorch.org/whl/cpu'
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $GpuList = & nvidia-smi -L 2>$null
  if ($LASTEXITCODE -eq 0 -and $GpuList) { $TorchIndex = 'https://download.pytorch.org/whl/cu130' }
}
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --timeout 180 --retries 8 --index-url $TorchIndex --extra-index-url https://pypi.org/simple 'torch==2.12.0' 'torchvision==0.27.0' 'numpy>=1.26,<2.0' 'Pillow>=10,<15'
if ($LASTEXITCODE -ne 0) { throw 'No se pudo instalar PyTorch y torchvision.' }
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias de ModelBuilder.' }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check encontró conflictos en .venv.' }
Write-Host '==> Dependencias JavaScript'
npm ci
Write-Host 'Instalación completada. Ejecuta .\run-windows.ps1 o npm run dev.'
