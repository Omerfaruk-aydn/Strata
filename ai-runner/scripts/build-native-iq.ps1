param(
  [Parameter(Mandatory = $true)]
  [string]$GgmlRoot,
  [string]$BuildDirectory = 'native/build-iq',
  [string]$GgmlLibrary = ''
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
  throw 'CMake is required to build the native IQ bridge.'
}

$root = (Resolve-Path $GgmlRoot).Path
$libraryOption = @()
if ($GgmlLibrary) {
  $library = (Resolve-Path $GgmlLibrary).Path
  $libraryOption = @("-DSTRATA_GGML_LIBRARY=$library")
}
cmake -S native -B $BuildDirectory -DSTRATA_GGML_ROOT="$root" -DSTRATA_ENABLE_CUDA=OFF @libraryOption
cmake --build $BuildDirectory --config Release --target strata_iq

Write-Host "Native IQ bridge built in $BuildDirectory. Set STRATA_IQ_LIBRARY to the resulting library." -ForegroundColor Green
