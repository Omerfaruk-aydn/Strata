cmake_minimum_required(VERSION 3.21)
project(strata_native LANGUAGES CXX)

option(STRATA_ENABLE_CUDA "Build the CUDA Strata backend" OFF)

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(strata_cuda SHARED cuda/strata_cuda.cu)
  set_target_properties(strata_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "strata_cuda"
  )
  target_compile_definitions(strata_cuda PRIVATE STRATA_CUDA_EXPORTS)
endif()
