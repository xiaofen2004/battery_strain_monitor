param([string]$OutputName = ('battery_strain_monitor_exe_' + (Get-Date -Format 'yyyyMMdd')))
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run setup.ps1 first, then install pyinstaller into .venv.' }
& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller missing. Run: .\.venv\Scripts\python.exe -m pip install pyinstaller==6.22.3' }
$distpath = Join-Path '..\outputs' $OutputName
& $python -m PyInstaller --noconfirm --clean --onefile --windowed --name BatteryStrainMonitor --distpath $distpath --workpath '.build-work' --specpath '.build-spec' --collect-data matplotlib --hidden-import matplotlib.backends.backend_tkagg --hidden-import matplotlib.backends.backend_pdf --hidden-import openpyxl --hidden-import xlrd --hidden-import pandas.io.excel._openpyxl exe_entry.py
if ($LASTEXITCODE -ne 0) { throw 'EXE build failed.' }
Write-Host ('EXE output: ' + (Resolve-Path -LiteralPath $distpath))
