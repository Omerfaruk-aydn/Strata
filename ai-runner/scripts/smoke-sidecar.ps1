param(
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$binary = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src-tauri\binaries') `
    -Filter 'ai_runner_backend-*.exe' | Select-Object -First 1
if (-not $binary) {
    throw 'Sidecar binary not found. Run npm run sidecar:build first.'
}

$listener = [System.Net.Sockets.TcpListener]::new(
    [System.Net.IPAddress]::Loopback,
    0
)
$listener.Start()
$port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()

$stdout = Join-Path $env:TEMP 'ai-runner-sidecar-smoke.stdout.log'
$stderr = Join-Path $env:TEMP 'ai-runner-sidecar-smoke.stderr.log'
Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue

$process = Start-Process `
    -FilePath $binary.FullName `
    -ArgumentList '--host', '127.0.0.1', '--port', $port `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
try {
    $root = $null
    while ((Get-Date) -lt $deadline -and -not $process.HasExited) {
        try {
            $root = Invoke-RestMethod -Uri "http://127.0.0.1:$port/" -TimeoutSec 2
            if ($root.status -eq 'running') {
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
        $process.Refresh()
    }

    if (-not $root -or $root.status -ne 'running') {
        $reason = if ($process.HasExited) { "exit code $($process.ExitCode)" } else { 'timeout' }
        throw "Sidecar did not become ready: $reason"
    }

    $status = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/status" -TimeoutSec 10
    if ($status.status -ne 'running') {
        throw 'Sidecar status endpoint returned an unexpected payload.'
    }

    Write-Output "Sidecar smoke test passed on port $port."
} finally {
    if (-not $process.HasExited) {
        & taskkill.exe /PID $process.Id /T /F | Out-Null
    }
}
