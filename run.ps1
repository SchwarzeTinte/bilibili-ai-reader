param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectDir "requirements.txt"
$RequirementsMarker = Join-Path $VenvDir ".requirements.sha256"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Stop-WithHelp([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    exit 1
}

function Get-ProjectStreamlitProcesses {
    $AppScript = Join-Path $ProjectDir "app.py"
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match '^python(?:w)?\.exe$' -and
                $_.CommandLine -and
                $_.CommandLine.IndexOf('streamlit run', [System.StringComparison]::OrdinalIgnoreCase) -ge 0 -and
                $_.CommandLine.IndexOf($AppScript, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
            }
    )
}

function Get-ListeningProjectPort($Processes) {
    $ProcessIds = @($Processes | Select-Object -ExpandProperty ProcessId)
    if (-not $ProcessIds) {
        return $null
    }
    $Listeners = @(
        Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
            Where-Object { $_.OwningProcess -in $ProcessIds } |
            Sort-Object @{ Expression = { if ($_.LocalPort -eq 8501) { 0 } else { 1 } } }, LocalPort
    )
    if ($Listeners) {
        return $Listeners[0].LocalPort
    }
    return $null
}

$ExistingProcesses = if ($CheckOnly) { @() } else { Get-ProjectStreamlitProcesses }
if ($ExistingProcesses) {
    $ExistingPort = Get-ListeningProjectPort $ExistingProcesses
    if (-not $ExistingPort) {
        # A concurrently launched instance may need a moment before it starts listening.
        for ($Attempt = 0; $Attempt -lt 10 -and -not $ExistingPort; $Attempt++) {
            Start-Sleep -Milliseconds 500
            $ExistingProcesses = Get-ProjectStreamlitProcesses
            $ExistingPort = Get-ListeningProjectPort $ExistingProcesses
        }
    }
    if ($ExistingPort) {
        $ExistingUrl = "http://localhost:$ExistingPort"
        Write-Host "Bilibili AI Reader is already running at $ExistingUrl" -ForegroundColor Green
        Start-Process $ExistingUrl
        exit 0
    }

    $Cutoff = (Get-Date).AddSeconds(-30)
    $OrphanIds = @(
        $ExistingProcesses |
            Where-Object { $_.CreationDate -lt $Cutoff } |
            Select-Object -ExpandProperty ProcessId
    )
    if ($OrphanIds) {
        Write-Host "Cleaning up an old Bilibili AI Reader process that no longer owns a port..." -ForegroundColor Yellow
        Stop-Process -Id $OrphanIds -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "Another Bilibili AI Reader instance is still starting. Please wait a moment." -ForegroundColor Yellow
        exit 0
    }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Stop-WithHelp "FFmpeg is missing. Install it with: winget install --id Gyan.FFmpeg"
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "Creating the Python virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $VenvDir
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $VenvDir
    } else {
        Stop-WithHelp "Python is missing. Install Python 3.11 or newer from https://www.python.org/downloads/"
    }
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path -LiteralPath $PythonExe))) {
        Stop-WithHelp "Python could not create .venv. Install Python 3.11+ and enable 'Add Python to PATH'."
    }
    & $PythonExe -m pip install --upgrade pip
}

& $PythonExe -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
if ($LASTEXITCODE -ne 0) {
    Stop-WithHelp "Python 3.10 or newer is required. Delete .venv after upgrading Python, then run again."
}

$RequirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $RequirementsFile).Hash
$InstalledHash = if (Test-Path -LiteralPath $RequirementsMarker) {
    (Get-Content -Raw -LiteralPath $RequirementsMarker).Trim()
} else {
    ""
}

& $PythonExe -c 'import streamlit, yt_dlp, faster_whisper, openai, anthropic, google.genai' 2>$null
$ImportsOk = $LASTEXITCODE -eq 0
if ((-not $ImportsOk) -or ($InstalledHash -ne $RequirementsHash)) {
    Write-Host "Installing or updating project dependencies..."
    & $PythonExe -m pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Stop-WithHelp "Dependency installation failed. Check the network output above and run this script again."
    }
    Set-Content -NoNewline -Encoding ASCII -LiteralPath $RequirementsMarker -Value $RequirementsHash
}

& $PythonExe -m pip check
if ($LASTEXITCODE -ne 0) {
    Stop-WithHelp "The Python environment has conflicting packages. Delete .venv and run again."
}

if ($CheckOnly) {
    Write-Host "Environment check passed." -ForegroundColor Green
    exit 0
}

Write-Host "Starting Bilibili AI Reader at http://localhost:8501"
& $PythonExe -m streamlit run (Join-Path $ProjectDir "app.py") --server.showEmailPrompt=false --browser.gatherUsageStats=false
