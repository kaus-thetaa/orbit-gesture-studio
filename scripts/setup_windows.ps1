# One-shot setup for Windows PowerShell. Run from the repo root:  .\scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw "Python launcher 'py' not found. Install Python 3.11/3.12 from python.org." }
py -3.11 -m venv .venv 2>$null; if ($LASTEXITCODE -ne 0) { py -3.12 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m orbit doctor
.\.venv\Scripts\python.exe -m orbit download-models
Write-Host "`nDone. Activate with .\.venv\Scripts\Activate.ps1, then run: python -m orbit" -ForegroundColor Green
