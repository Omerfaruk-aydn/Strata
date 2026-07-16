# Strata native backends

The Python runtime remains the correctness reference. Native backends are
optional, capability-detected components and are never assumed to exist.

## CUDA

The CUDA backend currently provides a packed `ternary-q05` matrix-vector
kernel through a small C ABI. Build it with CUDA Toolkit and CMake:

```powershell
cmake -S native -B native/build -DSTRATA_ENABLE_CUDA=ON
cmake --build native/build --config Release
```

Set `STRATA_CUDA_LIBRARY` to the resulting shared library when it is outside
the standard search paths. The API reports the backend under
`/api/ultra/capabilities`; selecting `cuda` without the library returns a
clear runtime error rather than silently executing on the CPU.

The kernel is deliberately narrow at this stage: it supports only the
`ternary-q05` weight codec and synchronous host-buffer calls. Sparse05,
batched streams, Vulkan, and fused transformer kernels require separate
benchmarked implementations and are not advertised as CUDA support yet.

## Native IQ bridge

If a local llama.cpp/ggml source checkout is available, configure
`STRATA_GGML_ROOT` when configuring CMake. The resulting `strata_iq` library
exposes a stable ABI for GGML IQ1/IQ2/IQ3 dequantization and can be selected
with `STRATA_IQ_LIBRARY`. The bridge validates type, block size, and element
count before calling the upstream dequantizer. Without this optional library,
IQ codecs remain unavailable and are reported as such by the API.
