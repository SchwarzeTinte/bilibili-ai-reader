$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppUrl = "http://localhost:8501"

try {
    $response = Invoke-WebRequest -Uri $AppUrl -UseBasicParsing -TimeoutSec 2
    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        Start-Process $AppUrl
        exit 0
    }
} catch {
    # The app is not running yet; start it below.
}

Set-Location -LiteralPath $ProjectDir
& (Join-Path $ProjectDir "run.ps1")
