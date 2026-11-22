param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Unable to create environment. Install Python with Tcl/Tk first." }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check network and retry." }
& .\.venv\Scripts\python.exe -c "import tkinter, numpy, scipy, pandas, matplotlib, openpyxl, xlrd, serial; print('Environment ready')"
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }
