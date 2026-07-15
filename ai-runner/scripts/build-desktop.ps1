param(
    [switch]$SkipSidecarBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $projectRoot
$localCargoBin = Join-Path $workspaceRoot '.tools\cargo\bin'

if (Test-Path -LiteralPath (Join-Path $localCargoBin 'cargo.exe')) {
    $env:PATH = "$localCargoBin;$env:PATH"
    $env:CARGO_HOME = Join-Path $workspaceRoot '.tools\cargo'
    $env:RUSTUP_HOME = Join-Path $workspaceRoot '.tools\rustup'
}

Push-Location $projectRoot
try {
    if ($SkipSidecarBuild) {
        $env:AI_RUNNER_SKIP_SIDECAR_BUILD = '1'
    }
    & npx.cmd tauri build
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri build başarısız oldu (çıkış kodu: $LASTEXITCODE)."
    }
} finally {
    Remove-Item Env:AI_RUNNER_SKIP_SIDECAR_BUILD -ErrorAction SilentlyContinue
    Pop-Location
}
