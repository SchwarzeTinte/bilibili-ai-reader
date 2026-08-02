$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppScript = Join-Path $ProjectDir "app.py"
$AllProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
$Roots = @(
    $AllProcesses | Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and
        $_.CommandLine -and
        $_.CommandLine.IndexOf($AppScript, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    }
)

if (-not $Roots) {
    Write-Host "Bilibili AI Reader is already stopped." -ForegroundColor Green
    exit 0
}

$Seen = [System.Collections.Generic.HashSet[int]]::new()
$StopOrder = [System.Collections.Generic.List[int]]::new()

function Add-ProcessTree([int]$ProcessId) {
    foreach ($Child in @($AllProcesses | Where-Object { $_.ParentProcessId -eq $ProcessId })) {
        Add-ProcessTree -ProcessId $Child.ProcessId
    }
    if ($Seen.Add($ProcessId)) {
        $StopOrder.Add($ProcessId)
    }
}

foreach ($Root in $Roots) {
    Add-ProcessTree -ProcessId $Root.ProcessId
}

foreach ($ProcessId in $StopOrder) {
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 500
$Remaining = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and
        $_.CommandLine -and
        $_.CommandLine.IndexOf($AppScript, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    }
)
if ($Remaining) {
    Write-Host "The app did not stop completely. Try running stop.bat as administrator." -ForegroundColor Red
    exit 1
}

Write-Host "Bilibili AI Reader and its background tasks have stopped." -ForegroundColor Green
