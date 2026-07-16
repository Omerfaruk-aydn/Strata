param(
  [string]$BuildDirectory = 'native/build-cuda',
  [string]$CudaArchitectures = ''
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
  throw 'CMake is required. Install CMake and retry.'
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  throw 'CUDA Toolkit is required. Install the NVIDIA CUDA Toolkit (including nvcc) and retry.'
}

if (-not $CudaArchitectures) {
  $query = nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
  if ($LASTEXITCODE -eq 0 -and $query) {
    $CudaArchitectures = (($query | Select-Object -First 1).Trim() -replace '\.', '')
  }
}

$configure = @('-S', 'native', '-B', $BuildDirectory, '-DSTRATA_ENABLE_CUDA=ON')
if ($CudaArchitectures) { $configure += "-DSTRATA_CUDA_ARCHITECTURES=$CudaArchitectures" }
cmake @configure
cmake --build $BuildDirectory --config Release --target strata_cuda
Write-Host "CUDA backend built successfully in $BuildDirectory" -ForegroundColor Green
