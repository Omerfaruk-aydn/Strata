$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed (exit code: $LASTEXITCODE)."
    }

    if ($env:AI_RUNNER_SKIP_SIDECAR_BUILD -ne '1') {
        & npm.cmd run sidecar:build
        if ($LASTEXITCODE -ne 0) {
            throw "Sidecar build failed (exit code: $LASTEXITCODE)."
        }
    }
} finally {
    Pop-Location
}
