$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating the Python virtual environment..."
    py -3 -m venv (Join-Path $ProjectDir ".venv")
    & $PythonExe -m pip install --upgrade pip
}

& $PythonExe -c 'import streamlit, yt_dlp, faster_whisper, openai, google.genai' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $PythonExe -m pip install -r (Join-Path $ProjectDir "requirements.txt")
}

& $PythonExe -m streamlit run (Join-Path $ProjectDir "app.py") --server.showEmailPrompt=false --browser.gatherUsageStats=false
