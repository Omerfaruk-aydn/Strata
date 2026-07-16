// Stable Strata IQ ABI over a native GGML build.
// The GGML library remains the format authority; this bridge only owns the
// type dispatch and ABI required by the Python converter.

#include <cstdint>
#include <cstddef>
#include <cmath>

#include "ggml-quants.h"

#if defined(_WIN32)
#define STRATA_IQ_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_IQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif

STRATA_IQ_EXPORT int strata_ggml_dequant_iq(
    std::uint32_t type_id,
    const std::uint8_t * raw,
    std::size_t raw_bytes,
    std::float_t * output,
    std::int64_t value_count) {
  if (!raw || !output || value_count <= 0) return 1;
  if (value_count % 256 != 0) return 2;

  switch (type_id) {
    case 16:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq2_xxs)) return 3;
      dequantize_row_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(raw), output, value_count);
      return 0;
    case 17:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq2_xs)) return 3;
      dequantize_row_iq2_xs(reinterpret_cast<const block_iq2_xs *>(raw), output, value_count);
      return 0;
    case 18:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq3_xxs)) return 3;
      dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(raw), output, value_count);
      return 0;
    case 21:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq3_s)) return 3;
      dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(raw), output, value_count);
      return 0;
    case 22:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq2_s)) return 3;
      dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s *>(raw), output, value_count);
      return 0;
    case 19:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq1_s)) return 3;
      dequantize_row_iq1_s(reinterpret_cast<const block_iq1_s *>(raw), output, value_count);
      return 0;
    case 29:
      if (raw_bytes != static_cast<std::size_t>(value_count / 256) * sizeof(block_iq1_m)) return 3;
      dequantize_row_iq1_m(reinterpret_cast<const block_iq1_m *>(raw), output, value_count);
      return 0;
    default:
      return 4;
  }
}
