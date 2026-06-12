$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "== GymPulse EXE Build =="
Write-Host "Project root: $ProjectRoot"

# Stop old running exe so dist folder is not locked.
Get-Process GymPulseWebUI -ErrorAction SilentlyContinue | Stop-Process -Force

# Create/use isolated Python 3.11 virtual environment.
$Python311 = py -3.11 -c "import sys; print(sys.executable)" 2>$null
if (-not $Python311) {
  throw "Python 3.11 64-bit is required. Install it, then run this script again."
}

if (Test-Path ".venv311\pyvenv.cfg") {
  $VenvHome = (Get-Content ".venv311\pyvenv.cfg" | Where-Object { $_ -match "^home\s*=" }) -replace "^home\s*=\s*", ""
  if (-not (Test-Path (Join-Path $VenvHome "python.exe"))) {
    Remove-Item -Recurse -Force ".venv311"
  }
}

py -3.11 -m venv .venv311

$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -r requirements-build.txt

# Clean previous output.
Remove-Item -Recurse -Force ".\build" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".\dist\GymPulseWebUI" -ErrorAction SilentlyContinue
Remove-Item -Force ".\GymPulseWebUI.spec" -ErrorAction SilentlyContinue

# Build onedir exe. On PyInstaller 6+, dependencies go into _internal by default.
& $Python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --windowed `
  --name GymPulseWebUI `
  --contents-directory _internal `
  --collect-all mindspore `
  --collect-all numpy `
  --collect-all pandas `
  --collect-all flask `
  --collect-all werkzeug `
  --add-data "pangu_dispatcher_core\mindspore_tabular_predictor.ckpt;pangu_dispatcher_core" `
  --add-data "pangu_dispatcher_core\templates;pangu_dispatcher_core\templates" `
  --hidden-import inference_engine `
  --hidden-import web_app `
  pangu_dispatcher_core\web_launcher.py

# Keep CSV editable beside the exe.
Copy-Item ".\field_dispatch_dataa.csv" ".\dist\GymPulseWebUI\field_dispatch_dataa.csv" -Force
Copy-Item ".\README_BUILD.txt" ".\dist\GymPulseWebUI\README.txt" -Force

Write-Host ""
Write-Host "BUILD COMPLETE"
Write-Host "Run this test:"
Write-Host "  .\dist\GymPulseWebUI\GymPulseWebUI.exe"
Write-Host ""
Write-Host "Transfer this whole folder to another PC:"
Write-Host "  $ProjectRoot\dist\GymPulseWebUI"
