// Strata CUDA backend: packed ternary-q0.5 matvec.
// The ABI is intentionally small so it can be loaded by ctypes/cffi without
// coupling the Python runtime to a particular CUDA or C++ standard library.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <limits>

#if defined(_WIN32)
#define STRATA_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

__global__ void kv_decode_kernel(
    const std::uint8_t* packed,
    const float* scales,
    float* output,
    std::uint32_t count,
    std::uint32_t group_size,
    std::uint32_t bits) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float scale = scales[index / group_size];
  if (bits == 1) {
    const std::uint8_t bit = (packed[index / 8] >> (index % 8)) & 1;
    output[index] = bit ? scale : -scale;
  } else {
    const std::uint8_t code = (packed[index / 4] >> ((index % 4) * 2)) & 3;
    output[index] = code == 0 ? 0.0f : code == 1 ? -scale : scale;
  }
}

__global__ void ternary_matvec_kernel(
    const std::uint8_t* packed,
    const float* scales,
    const float* vector,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t group_size) {
  const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  float total = 0.0f;
  const std::uint64_t base = static_cast<std::uint64_t>(row) * cols;
  for (std::uint32_t col = 0; col < cols; ++col) {
    const std::uint64_t index = base + col;
    const std::uint8_t code = (packed[index / 4] >> ((index % 4) * 2)) & 0x3;
    if (code != 0) {
      const float sign = code == 1 ? -1.0f : 1.0f;
      total += sign * scales[index / group_size] * vector[col];
    }
  }
  output[row] = total;
}

}  // namespace

// Returns a CUDA error code. Inputs and output are host pointers; the function
// owns temporary device buffers and is synchronous by design for a stable ABI.
STRATA_EXPORT int strata_cuda_ternary_matvec(
    const std::uint8_t* packed,
    const float* scales,
    const float* vector,
    float* output,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t group_size) {
  if (!packed || !scales || !vector || !output || rows == 0 || cols == 0 || group_size == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (static_cast<std::size_t>(rows) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(cols)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  std::uint8_t* d_packed = nullptr;
  float *d_scales = nullptr, *d_vector = nullptr, *d_output = nullptr;
  const std::size_t packed_bytes = (static_cast<std::size_t>(rows) * cols + 3) / 4;
  const std::size_t scale_count =
      (static_cast<std::size_t>(rows) * cols + group_size - 1) / group_size;
  cudaError_t status = cudaSuccess;
  status = cudaMalloc(&d_packed, packed_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_scales, scale_count * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_vector, static_cast<std::size_t>(cols) * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_output, static_cast<std::size_t>(rows) * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpy(d_packed, packed, packed_bytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpy(d_scales, scales, scale_count * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpy(d_vector, vector, static_cast<std::size_t>(cols) * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) goto cleanup;

  ternary_matvec_kernel<<<(rows + 255) / 256, 256>>>(
      d_packed, d_scales, d_vector, d_output, rows, cols, group_size);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaMemcpy(output, d_output, static_cast<std::size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost);
  }

cleanup:
  cudaFree(d_packed);
  cudaFree(d_scales);
  cudaFree(d_vector);
  cudaFree(d_output);
  return static_cast<int>(status);
}

STRATA_EXPORT int strata_cuda_kv_decode(
    const std::uint8_t* packed,
    const float* scales,
    float* output,
    std::uint32_t count,
    std::uint32_t group_size,
    std::uint32_t bits) {
  if (!packed || !scales || !output || count == 0 || group_size == 0 || (bits != 1 && bits != 2)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t payload_bytes = (static_cast<std::size_t>(count) + (bits == 1 ? 7 : 3)) / (bits == 1 ? 8 : 4);
  const std::size_t scale_count = (static_cast<std::size_t>(count) + group_size - 1) / group_size;
  std::uint8_t* d_packed = nullptr;
  float *d_scales = nullptr, *d_output = nullptr;
  cudaError_t status = cudaMalloc(&d_packed, payload_bytes);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_scales, scale_count * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMalloc(&d_output, static_cast<std::size_t>(count) * sizeof(float));
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpy(d_packed, packed, payload_bytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) goto cleanup;
  status = cudaMemcpy(d_scales, scales, scale_count * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) goto cleanup;
  kv_decode_kernel<<<(count + 255) / 256, 256>>>(d_packed, d_scales, d_output, count, group_size, bits);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) {
    status = cudaMemcpy(output, d_output, static_cast<std::size_t>(count) * sizeof(float), cudaMemcpyDeviceToHost);
  }
cleanup:
  cudaFree(d_packed);
  cudaFree(d_scales);
  cudaFree(d_output);
  return static_cast<int>(status);
}
