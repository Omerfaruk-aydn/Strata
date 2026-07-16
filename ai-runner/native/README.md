# Strata native backends

The Python runtime remains the correctness reference. Native backends are
optional, capability-detected components and are never assumed to exist.

Run `powershell -ExecutionPolicy Bypass -File scripts/check-native-toolchain.ps1`
before configuring a native build to see which tools are available.

## Windows prerequisites

Install CMake, a C++17 compiler (Visual Studio Build Tools or MinGW), and the
CUDA Toolkit only when enabling the CUDA backend. For the IQ bridge, also keep
a matching llama.cpp checkout with its GGML library built. Then run the
preflight script again before invoking `build-native-iq.ps1`.

## CUDA

The CUDA backend currently provides a packed `ternary-q05` matrix-vector
kernel through a small C ABI. Build it with CUDA Toolkit and CMake:

```powershell
cmake -S native -B native/build -DSTRATA_ENABLE_CUDA=ON
cmake --build native/build --config Release
```

For a reproducible deployment build, set the GPU architecture explicitly,
for example `-DSTRATA_CUDA_ARCHITECTURES=86;89`. Leaving it empty lets CMake
choose its default architecture policy.

The repository also includes an automatic Windows helper. It checks for
`nvcc`, detects the first GPU compute capability through `nvidia-smi`, and
builds the Release DLL:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-native-cuda.ps1
```

Set `STRATA_CUDA_LIBRARY` to the resulting shared library when it is outside
the standard search paths. The API reports the backend under
`/api/ultra/capabilities`; selecting `cuda` without the library returns a
clear runtime error rather than silently executing on the CPU.

The native CUDA ABI supports the `ternary-q05` weight matvec and GPU decode of
the fixed-width `sign1` and `ternary05` KV profiles. The Python runtime selects
those KV kernels with `backend=auto` when the DLL is available and falls back
to the reference decoder otherwise. `sparse05`, batched streams, Vulkan, and
fused transformer kernels require separate benchmarked implementations and
are not advertised as CUDA support yet.

## Native IQ bridge

If a local llama.cpp/ggml source checkout is available, configure
`STRATA_GGML_ROOT` when configuring CMake. The resulting `strata_iq` library
exposes a stable ABI for GGML IQ1/IQ2/IQ3/IQ4_XS dequantization and can be selected
with `STRATA_IQ_LIBRARY`. The bridge validates type, block size, and element
count before calling the upstream dequantizer. Without this optional library,
IQ codecs remain unavailable and are reported as such by the API.

On Windows, the repository helper can be used after installing CMake:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-native-iq.ps1 `
  -GgmlRoot C:\src\llama.cpp
```

If the GGML library is in a non-standard location, pass it explicitly with
`-GgmlLibrary C:\src\llama.cpp\build\src\Release\ggml-base.lib`.

When the bridge DLL depends on a separately built `ggml-base.dll`, expose its
directory at runtime with `STRATA_GGML_RUNTIME_DIR`. This is especially useful
on Windows, where the Python loader registers the directory explicitly:

```powershell
$env:STRATA_IQ_LIBRARY = 'C:\src\Strata\native\build-iq\Release\strata_iq.dll'
$env:STRATA_GGML_RUNTIME_DIR = 'C:\src\llama.cpp\build\bin\Release'
```

The CMake integration accepts both the legacy `src` layout and the current
llama.cpp layout (`ggml/src` plus `ggml/include`).

The GGML checkout and the built `ggml` library must match the GGUF type layout
used by the model. Strata does not copy or reinterpret upstream codebooks, and
it does not mark IQ support active until the shared library loads successfully.
