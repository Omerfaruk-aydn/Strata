param(
    [switch]$SkipToolInstall,
    [string]$SourcePython = 'python'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.tools\sidecar-venv'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$distRoot = Join-Path $projectRoot 'dist-sidecar'
$buildRoot = Join-Path $projectRoot 'build-sidecar'
$binaryRoot = Join-Path $projectRoot 'src-tauri\binaries'
$workspaceRoot = Split-Path -Parent $projectRoot
$localCargoHome = Join-Path $workspaceRoot '.tools\cargo'
$localRustupHome = Join-Path $workspaceRoot '.tools\rustup'
$localRustc = Join-Path $localCargoHome 'bin\rustc.exe'

if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $SourcePython -m venv --system-site-packages $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Sidecar build environment could not be created (exit code: $LASTEXITCODE)."
    }
}

# Make packages from the caller's active Python environment visible to the
# isolated PyInstaller tool environment. This preserves a deliberately chosen
# CUDA/Metal llama-cpp-python wheel instead of silently replacing it.
$sourceSitePackages = & $SourcePython -c "import site; print('\n'.join(site.getsitepackages()))"
$toolSitePackages = & $pythonExe -c "import site; print(site.getsitepackages()[0])"
$sourcePaths = @($sourceSitePackages | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Sort-Object -Unique)
[System.IO.File]::WriteAllLines(
    (Join-Path $toolSitePackages 'ai_runner_source_environment.pth'),
    $sourcePaths
)

if (-not $SkipToolInstall) {
    & $pythonExe -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'backend\requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime dependency installation failed (exit code: $LASTEXITCODE)."
    }
    & $pythonExe -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'requirements-build.txt')
    if ($LASTEXITCODE -ne 0) {
        throw "Build dependency installation failed (exit code: $LASTEXITCODE)."
    }
}

$targetTriple = 'x86_64-pc-windows-msvc'
if (Test-Path -LiteralPath $localRustc) {
    $env:CARGO_HOME = $localCargoHome
    $env:RUSTUP_HOME = $localRustupHome
    $env:PATH = "$(Split-Path -Parent $localRustc);$env:PATH"
}
$rustc = Get-Command rustc -ErrorAction SilentlyContinue
if ($rustc) {
    $rustcPath = if ($rustc.Source) { $rustc.Source } else { $rustc.FullName }
    $hostLine = & $rustcPath -vV | Where-Object { $_ -like 'host:*' } | Select-Object -First 1
    if ($hostLine) {
        $targetTriple = ($hostLine -replace '^host:\s*', '').Trim()
    }
}

New-Item -ItemType Directory -Force -Path $binaryRoot | Out-Null

# llama-cpp-python's CUDA wheel loads NVIDIA runtime DLLs dynamically. Make
# those directories available during analysis and preserve their package
# layout so backend.main can discover them in the frozen application.
$nvidiaBinaryArgs = @()
$sitePackageRoots = & $pythonExe -c "import sys; print('\n'.join(p for p in sys.path if p.lower().endswith('site-packages')))"
foreach ($sitePackageRoot in $sitePackageRoots) {
    $nvidiaRoot = Join-Path $sitePackageRoot 'nvidia'
    if (-not (Test-Path -LiteralPath $nvidiaRoot)) {
        continue
    }

    foreach ($packageDir in Get-ChildItem -LiteralPath $nvidiaRoot -Directory) {
        if ($packageDir.Name -notin @('cublas', 'cuda_runtime')) {
            continue
        }
        $binDir = Join-Path $packageDir.FullName 'bin'
        if (-not (Test-Path -LiteralPath $binDir)) {
            continue
        }

        $dlls = Get-ChildItem -LiteralPath $binDir -Filter '*.dll' -File
        if (-not $dlls) {
            continue
        }

        $env:PATH = "$binDir;$env:PATH"
        foreach ($dll in $dlls) {
            $nvidiaBinaryArgs += '--add-binary'
            $nvidiaBinaryArgs += ($dll.FullName + ";nvidia\$($packageDir.Name)\bin")
        }
    }
}

$pyInstallerArgs = @(
    '-m', 'PyInstaller',
    '--noconfirm',
    '--clean',
    '--onefile',
    '--name', 'ai_runner_backend',
    '--distpath', $distRoot,
    '--workpath', $buildRoot,
    '--specpath', $buildRoot,
    '--paths', $projectRoot,
    '--collect-binaries', 'llama_cpp',
    '--collect-data', 'llama_cpp',
    '--hidden-import', 'llama_cpp.llama_speculative',
    '--add-data', ((Join-Path $projectRoot 'backend\db\schema.sql') + ';backend\db'),
    '--exclude-module', 'backend.tests',
    '--exclude-module', 'llama_cpp.server',
    '--exclude-module', 'pytest',
    '--exclude-module', 'openai',
    '--exclude-module', 'mcp',
    '--exclude-module', 'boto3',
    '--exclude-module', 'botocore',
    '--exclude-module', 's3transfer'
)
$pyInstallerArgs += $nvidiaBinaryArgs
$pyInstallerArgs += (Join-Path $projectRoot 'backend_sidecar.py')

& $pythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed (exit code: $LASTEXITCODE)."
}

$warningFile = Join-Path $buildRoot 'ai_runner_backend\warn-ai_runner_backend.txt'
if (Test-Path -LiteralPath $warningFile) {
    $missingLibraries = Select-String -LiteralPath $warningFile -Pattern 'Library not found:'
    if ($missingLibraries) {
        $details = ($missingLibraries.Line | Sort-Object -Unique) -join [Environment]::NewLine
        throw "Sidecar has unresolved native libraries:$([Environment]::NewLine)$details"
    }
}

$builtBinary = Join-Path $distRoot 'ai_runner_backend.exe'
if (-not (Test-Path -LiteralPath $builtBinary)) {
    throw "Sidecar binary was not created: $builtBinary"
}

$targetBinary = Join-Path $binaryRoot "ai_runner_backend-$targetTriple.exe"
Copy-Item -LiteralPath $builtBinary -Destination $targetBinary -Force
Write-Output "Sidecar ready: $targetBinary"
