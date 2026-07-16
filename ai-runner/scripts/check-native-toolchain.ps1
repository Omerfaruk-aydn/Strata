$ErrorActionPreference = 'Stop'

$checks = @(
  @{ Name = 'cmake'; Required = $true },
  @{ Name = 'cl'; Required = $false },
  @{ Name = 'g++'; Required = $false },
  @{ Name = 'nvcc'; Required = $false },
  @{ Name = 'ninja'; Required = $false }
)

$missingRequired = @()
foreach ($check in $checks) {
  $command = Get-Command $check.Name -ErrorAction SilentlyContinue
  if ($command) {
    Write-Output ("{0}: available ({1})" -f $check.Name, $command.Source)
  } else {
    Write-Output ("{0}: missing{1}" -f $check.Name, $(if ($check.Required) { ' [required]' } else { '' }))
    if ($check.Required) { $missingRequired += $check.Name }
  }
}

if ($missingRequired.Count -gt 0) {
  Write-Output ("ERROR: Native build cannot start; install: {0}" -f ($missingRequired -join ', '))
  exit 2
}

Write-Output 'Native CMake configuration can start. CUDA is optional and required only for STRATA_ENABLE_CUDA=ON.'
