cmake_minimum_required(VERSION 3.21)
project(strata_native LANGUAGES CXX)

option(STRATA_ENABLE_CUDA "Build the CUDA Strata backend" OFF)
set(STRATA_GGML_ROOT "" CACHE PATH "Path to a llama.cpp ggml source checkout")
set(STRATA_GGML_LIBRARY "" CACHE FILEPATH "Explicit path to the GGML library for the IQ bridge")

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

if(STRATA_GGML_ROOT)
  if(NOT STRATA_GGML_LIBRARY)
    find_library(STRATA_GGML_LIBRARY
      NAMES ggml ggml-base libggml libggml-base
      PATHS
        "${STRATA_GGML_ROOT}/build/src"
        "${STRATA_GGML_ROOT}/build/src/Release"
        "${STRATA_GGML_ROOT}/build/bin"
        "${STRATA_GGML_ROOT}/build/bin/Release"
      REQUIRED
    )
  endif()
  if(NOT EXISTS "${STRATA_GGML_LIBRARY}")
    message(FATAL_ERROR "STRATA_GGML_LIBRARY does not exist: ${STRATA_GGML_LIBRARY}")
  endif()
  add_library(strata_iq SHARED iq/strata_iq.cpp)
  set(STRATA_GGML_INCLUDE_DIRS
    "${STRATA_GGML_ROOT}/include"
    "${STRATA_GGML_ROOT}/src"
  )
  if(EXISTS "${STRATA_GGML_ROOT}/ggml/src/ggml-quants.h")
    list(APPEND STRATA_GGML_INCLUDE_DIRS "${STRATA_GGML_ROOT}/ggml/src")
    if(EXISTS "${STRATA_GGML_ROOT}/ggml/include/ggml.h")
      list(APPEND STRATA_GGML_INCLUDE_DIRS "${STRATA_GGML_ROOT}/ggml/include")
    endif()
  endif()
  target_include_directories(strata_iq PRIVATE
    ${STRATA_GGML_INCLUDE_DIRS}
  )
  target_link_libraries(strata_iq PRIVATE "${STRATA_GGML_LIBRARY}")
  set_target_properties(strata_iq PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "strata_iq"
  )
endif()
