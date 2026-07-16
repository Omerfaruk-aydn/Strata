param(
  [Parameter(Mandatory = $true)]
  [string]$GgmlRoot,
  [string]$BuildDirectory = 'native/build-iq'
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
  throw 'CMake is required to build the native IQ bridge.'
}

$root = (Resolve-Path $GgmlRoot).Path
cmake -S native -B $BuildDirectory -DSTRATA_GGML_ROOT="$root" -DSTRATA_ENABLE_CUDA=OFF
cmake --build $BuildDirectory --config Release --target strata_iq

Write-Host "Native IQ bridge built in $BuildDirectory. Set STRATA_IQ_LIBRARY to the resulting library." -ForegroundColor Green
