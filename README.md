# AI Runner

> A production-oriented local GGUF model runner for Windows, built with React, FastAPI, llama.cpp, and Tauri.

[![CI](https://github.com/Omerfaruk-aydn/Strata/actions/workflows/ci.yml/badge.svg)](https://github.com/Omerfaruk-aydn/Strata/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](ai-runner/LICENSE)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D4.svg)](https://www.microsoft.com/windows)

> The implementation is located in [`ai-runner/`](ai-runner/). Run the development and build commands from that directory.

AI Runner lets you discover, download, configure, and run GGUF language models locally. It combines a React desktop interface with an authenticated FastAPI service and a Tauri 2 shell. After a model is available locally, inference, prompts, completions, sessions, and settings remain on the machine.

## Vision

AI Runner exists to make large language models practical on ordinary machines.
Its purpose is to squeeze the most capability out of limited VRAM, RAM, and CPU budgets so that people can run serious AI locally without needing datacenter hardware.
The project focuses on one core idea: give every user a realistic path to useful, high-performance on-device AI, even when the machine is constrained.

## Innovation Direction

The next wave of improvement is not just "smaller models" but smarter model execution.
The most important areas to push forward are:

- Better quantization strategies that preserve reasoning quality at lower bit widths.
- More accurate capacity planning so the app can predict fit before a load attempt.
- Smarter layer placement across GPU, CPU, and file-backed memory pressure.
- Adaptive context compression that keeps conversations usable without wasting memory.
- Runtime-aware backend selection that automatically chooses the best engine for the hardware.
- Measured optimization loops that learn from benchmark results instead of relying only on estimates.
- Safer managed re-quantization workflows for rebuilding local models into more efficient formats.
- Multi-GPU balancing that treats uneven VRAM as a first-class scheduling problem.

If the project keeps evolving in that direction, it can become more than a local runner: it can become a practical execution layer for high-end AI on modest hardware.

## Highlights

- Local-first inference using llama.cpp and GGUF models.
- Hugging Face search with exact quantization selection.
- Resumable HTTP Range downloads with progress, speed, ETA, cancellation, and partial-file recovery.
- GGUF magic-byte validation, optional SHA-256 verification, safe filenames, and disk/cache limits.
- GPU layer planning, selected GPU support, multi-GPU tensor splitting, CPU thread controls, mmap, mlock, Flash Attention, and KV-cache quantization.
- Extreme Model Center for 70B–200B feasibility analysis, capacity presets, adaptive OOM recovery, measured benchmarking, and hardware-specific known-good profiles.
- Native backend capability detection for CUDA, Vulkan, Metal, SYCL, and CPU builds, plus managed llama.cpp quantization jobs when llama-quantize is installed.
- Streaming chat with interrupt support, persisted SQLite sessions, Markdown/JSON export, and partial-response recovery.
- OpenAI-compatible model and chat endpoints.
- Live GPU, VRAM, RAM, TTFT, token-speed, and generation telemetry over WebSocket.
- Optional Bearer/API-key authentication, origin validation, secret redaction, and explicit consent for network exposure.
- Tauri 2 Windows desktop packaging with a PyInstaller backend sidecar, MSI, and NSIS installers.

## Architecture

~~~mermaid
flowchart LR
    UI[React + Zustand UI] -->|HTTP / SSE / WebSocket| API[FastAPI service]
    API --> DB[(SQLite)]
    API --> MM[Model manager]
    MM --> HF[Hugging Face Hub]
    API --> ENG[Inference engine]
    ENG --> LLAMA[llama.cpp / GGUF]
    API --> HW[Hardware and telemetry]
    Tauri[Tauri desktop shell] -->|supervises| Sidecar[Python backend sidecar]
    Tauri --> UI
~~~

| Layer | Responsibility | Technologies |
|---|---|---|
| Presentation | Chat, model shelf, settings, optimizer, telemetry | React, Vite, Zustand |
| API | Authenticated HTTP, SSE, WebSocket, validation | FastAPI, Pydantic, Uvicorn |
| Runtime | Model loading, generation, memory planning | llama.cpp, GGUF |
| Persistence | Sessions, messages, settings | SQLite, aiosqlite |
| Desktop | Window, sidecar lifecycle, cleanup, packaging | Tauri 2, Rust |
| Distribution | Frozen Python runtime and native libraries | PyInstaller, PowerShell |

## Requirements

### Runtime

- Windows 10/11 for the packaged desktop build.
- Python 3.11 or 3.12.
- Node.js 20 or newer and npm 10 or newer.
- Enough RAM and disk space for the selected model; 16 GB RAM is a practical starting point.
- Optional NVIDIA GPU with a compatible CUDA-enabled llama-cpp-python wheel.

### Desktop packaging

- Rust stable with the MSVC toolchain.
- Microsoft C++ Build Tools.
- WebView2 runtime.
- PowerShell.
- A Python environment containing the intended CPU/CUDA-compatible llama-cpp-python wheel.

Pinned runtime dependencies are in [ai-runner/backend/requirements.txt](ai-runner/backend/requirements.txt). Development and CI dependencies are separated into [ai-runner/backend/requirements-dev.txt](ai-runner/backend/requirements-dev.txt) and [ai-runner/backend/requirements-ci.txt](ai-runner/backend/requirements-ci.txt).

## Quick start

Open PowerShell in the ai-runner directory:

~~~powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\\requirements-dev.txt
npm ci
~~~

Start the backend and frontend in separate terminals:

~~~powershell
# Terminal 1
python -m backend.main
~~~

~~~powershell
# Terminal 2
npm run dev
~~~

Default URLs:

| Service | URL |
|---|---|
| Web UI | http://localhost:5173 |
| API | http://127.0.0.1:8420 |
| Swagger UI | http://127.0.0.1:8420/docs |
| OpenAPI JSON | http://127.0.0.1:8420/openapi.json |

## Model workflow

1. Open Model Shelf and search for a GGUF repository or model name.
2. Select a concrete single-file quantization. Q4_K_M is a good general starting point when available.
3. Review the size and available disk space.
4. Download the model. Interrupted downloads can be resumed.
5. Load the model and tune context length, GPU layers, batch size, threads, and optimization settings.
6. Start a chat session.

Models are stored by default under:

~~~text
%USERPROFILE%\\.ai-runner\\models\\
~~~

Downloads use temporary *.gguf.part files until validation succeeds. The model directory and cache limit can be changed from Settings.

## Extreme Model Mode

Extreme Model Mode is designed for models that do not fit entirely in VRAM. Open **Extreme Mode** from the header or press Ctrl+Shift+E.

The capacity engine:

1. Reads the actual GGUF architecture, parameter count, layer count, attention heads, KV heads, embedding size, and native context limit.
2. Detects the active native llama.cpp backend instead of assuming that an installed GPU is usable.
3. Separately budgets model weights, KV cache, compute buffers, VRAM reserve, RAM reserve, and physical working-set shortfall.
4. Produces an auditable GPU/CPU layer plan and distinguishes RAM-resident execution from file-backed mmap pressure.
5. Combines safe VRAM capacity across multiple GPUs and derives or validates tensor-split proportions.
6. Loads with bounded OOM recovery. Between attempts it can disable mlock, reduce GPU layers, reduce batch size, and reduce context size.
7. Persists the successful configuration against a model and hardware fingerprint.
8. Runs an on-device benchmark to replace planning estimates with measured token speed, TTFT, RAM, and system-wide VRAM data.

The built-in profiles are:

| Profile | Primary goal | Default context cap | Default batch |
|---|---|---:|---:|
| Safe | Maximum safety margin | 2,048 | 128 |
| Balanced | Reliability and usable throughput | 4,096 | 256 |
| Performance | Throughput when memory is comfortable | 8,192 | 512 |
| Maximum Capacity | Load the largest practical model | 2,048 | 64 |

For a 100B model on a 20 GB VRAM PC, physical system RAM remains important. AI Runner can split weights between GPU and CPU and use file-backed mmap under pressure, but it cannot make 100B weights physically fit inside 20 GB. Q3/IQ3-class GGUF files commonly require tens of gigabytes beyond VRAM; 64 GB RAM is a practical lower bound for many configurations and 128 GB provides substantially more headroom.

Adaptive rebalancing is performed only between generations by unloading and recreating the llama.cpp context. Transformer layers are never moved while a token generation is active.

Managed quantization uses the official llama.cpp companion executable when it is available. Point AI Runner at a trusted binary before startup:

~~~powershell
$env:AI_RUNNER_LLAMA_QUANTIZE = "C:\path\to\llama-quantize.exe"
python -m backend.main
~~~

Quantization runs as a single managed background job, validates both source and output GGUF structure, cleans partial output after failure/cancellation, and registers completed outputs in the local model shelf.

## Strata Ultra (experimental)

Strata Ultra is the independent low-bit research runtime inside the project. It is designed to explore how very large models can be stored and executed on constrained systems without making unsupported claims about quality or hardware limits.

Current capabilities:

- Versioned, checksummed `.strata` tensor containers.
- Experimental `STRATA-Q0.5` ternary tensor packing with per-group scales.
- Experimental `sparse05` variable-length codec that omits zero weights and can reach sub-bit storage on sparse groups.
- GGUF conversion for F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, and Q6_K source tensors.
- Independent reference CPU executor with on-the-fly dequantization.
- Pager-backed linear graphs, low-bit attention, SwiGLU MLP layers, and multi-block transformer execution.
- Automatic `python`/`numpy` execution backend selection with a correctness fallback.
- LRU layer paging with a hard byte budget.
- Runtime-managed `sign1` KV cache and experimental `ternary05` cache with sliding-window eviction.
- API endpoints for capabilities, memory estimates, benchmarks, paging plans, and local conversion.

The current reference executor is a correctness and format-validation milestone. It is not yet a complete tokenizer-backed conversational implementation, and a `.strata` file cannot be loaded by the existing llama.cpp inference path. IQ1, IQ2, and other unsupported block formats require dedicated decoders before conversion is enabled for them. The experimental ultra-low-bit modes intentionally trade output quality for minimum memory use and must be benchmarked against the original model.

Useful API calls:

~~~text
POST /api/ultra/memory
POST /api/ultra/benchmark
POST /api/ultra/paging-plan
POST /api/ultra/convert/{model_id}
POST /api/ultra/attention/step
POST /api/ultra/transformer/step
POST /api/ultra/graph/run
POST /api/ultra/generate
~~~

The runtime source is under [`ai-runner/backend/core/strata_ultra/`](ai-runner/backend/core/strata_ultra/). Every new codec and runtime component is covered by focused tests before it is considered ready for integration.

## API

The complete interactive API reference is available at http://127.0.0.1:8420/docs.

| Group | Endpoints | Purpose |
|---|---|---|
| Health | GET /, GET /api/status | Runtime health checks |
| Models | /api/models/* | Search, plan, download, progress, load, unload, list, delete |
| Chat | POST /api/chat, POST /api/chat/stop | Streaming generation and interruption |
| Sessions | /api/sessions/* | Create, list, update, delete, export |
| Settings | /api/settings/* | Read, validate, update, import, export |
| Hardware | /api/hardware/* | Hardware profile and refresh |
| Optimizer | /api/optimizer/* | Optimization actions and status |
| Extreme models | /api/extreme/* | Feasibility, capabilities, benchmark, rebalance, profiles, quantization |
| OpenAI compatibility | GET /v1/models, POST /v1/chat/completions | External client integration |
| Telemetry | WS /ws/telemetry | Live hardware and generation metrics |

### OpenAI-compatible example

Load a local model from the UI first:

~~~bash
curl http://127.0.0.1:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Explain GGUF in one paragraph."}],
    "max_tokens": 256,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
~~~

POST /api/chat returns an SSE stream with token events and a final completion event containing usage and finish metadata. POST /api/chat/stop interrupts the active generation for a session.

Telemetry is available at ws://127.0.0.1:8420/ws/telemetry. When authentication is enabled, the WebSocket key is supplied through the supported ai-runner subprotocol handshake instead of a query string.

## Security model

AI Runner is local-first, but the API can be exposed deliberately for trusted-network integrations.

- The default bind address is 127.0.0.1.
- Non-loopback binding requires explicit network consent and a non-empty API key.
- API keys can come from persisted settings or AI_RUNNER_API_KEY.
- Settings responses and settings export never return API keys in plaintext.
- API access supports Bearer and X-API-Key authentication.
- Origins are validated against trusted local origins and configured access rules.
- WebSocket authentication avoids putting secrets in URLs.
- Model paths are normalized and protected against traversal.
- GGUF files are validated before partial downloads are promoted.
- Tauri uses a restrictive Content Security Policy and supervises sidecar cleanup.

Example for a trusted local network:

~~~powershell
$env:AI_RUNNER_API_HOST = "0.0.0.0"
$env:AI_RUNNER_API_PORT = "8420"
$env:AI_RUNNER_API_KEY = "replace-with-a-long-random-secret"
python -m backend.main --allow-network
~~~

Do not expose the service directly to the public internet. Use a VPN, firewall, or authenticated reverse proxy for remote access.

## Configuration

Settings are persisted in SQLite and validated through an allowlist.

| Setting | Default | Description |
|---|---:|---|
| api_host | 127.0.0.1 | API bind address |
| api_port | 8420 | API port |
| api_key | unset | Optional API authentication secret |
| allow_network_access | false | Consent for non-loopback binding |
| model_dir | %USERPROFILE%\\.ai-runner\\models | GGUF storage directory |
| cache_size_limit_gb | 50 | Maximum model/cache footprint |
| max_context_length | 4096 | Prompt context budget |
| max_history_messages | 20 | Retained chat history |
| auto_context_prune | true | Automatic context trimming |
| n_threads | automatic | CPU inference threads |
| n_batch | 512 | llama.cpp batch size |
| use_mmap | true | Memory-map model files |
| use_mlock | true | Keep model pages resident when supported |
| flash_attn | true | Enable Flash Attention when supported |
| kv_cache_type | q4_0 | KV cache quantization |
| selected_gpu_index | 0 | Primary GPU index |
| tensor_split | unset | Multi-GPU split proportions |
| extreme_mode_enabled | true | Use metadata-aware capacity planning when GPU layers are automatic |
| extreme_preset | maximum_capacity | Default large-model capacity profile |
| adaptive_load | true | Retry memory-related load failures with safer settings |
| adaptive_max_attempts | 6 | Maximum bounded native load attempts |
| backend_preference | auto | Validate or force the native compute backend |
| context_compaction_mode | extractive_summary | Compress dropped history or remove oldest messages |

Local runtime data, logs, the SQLite database, model cache, and downloaded models live under %USERPROFILE%\\.ai-runner\\.

## Desktop packaging

~~~powershell
# Build the PyInstaller backend and native runtime bundle
npm run sidecar:build

# Start the frozen sidecar and verify /api/status
npm run sidecar:smoke

# Build frontend, sidecar, Tauri executable, MSI, and NSIS installer
npm run desktop:build
~~~

Installers are written to:

~~~text
src-tauri\\target\\release\\bundle\\msi\\
src-tauri\\target\\release\\bundle\\nsis\\
~~~

The sidecar build fails when required native libraries cannot be resolved. This prevents an installer from being produced when the packaged backend would not start. Models, caches, generated installers, and frozen sidecar binaries are intentionally excluded from Git.

## Development commands

~~~powershell
npm run dev
npm run build
npm run preview
npm run sidecar:build
npm run sidecar:smoke
npm run desktop:build
~~~

Backend quality checks:

~~~powershell
python -m pytest backend\\tests -q --cov=backend --cov-config=.coveragerc --cov-report=term-missing
python -m compileall -q backend backend_sidecar.py
python -m pip check
~~~

Rust/Tauri checks:

~~~powershell
cargo fmt --manifest-path src-tauri\\Cargo.toml -- --check
cargo check --release --manifest-path src-tauri\\Cargo.toml --locked
~~~

Dependency audit:

~~~powershell
npm audit --audit-level=moderate
~~~

The repository CI workflow runs the frontend build, backend tests with the coverage gate, dependency validation, and Rust checks on Windows.

## Validation status

- 214 backend tests passed.
- 77.45% backend source coverage.
- 70% minimum coverage gate.
- 0 npm audit vulnerabilities.
- No known vulnerabilities in the pinned Python dependency audit.
- Tauri release check passed.
- Packaged desktop health-check passed.

## Project layout

~~~text
Strata/
├── .github/workflows/ci.yml       Windows CI workflow
├── ai-runner/
│   ├── backend/
│   │   ├── api/                   FastAPI routes, auth, SSE, WebSocket
│   │   ├── core/                  Inference, hardware, memory, optimization
│   │   ├── db/                    SQLite schema and async persistence
│   │   ├── models/                Hugging Face and local model management
│   │   └── tests/                 Unit and integration tests
│   ├── scripts/                   Sidecar, smoke, and desktop build scripts
│   ├── src/                       React UI and Zustand stores
│   ├── src-tauri/                 Rust desktop shell and bundle metadata
│   ├── backend_sidecar.py         Frozen backend entry point
│   ├── baslat_0_vram.bat          Windows convenience launcher
│   ├── package.json               Frontend and desktop commands
│   └── README.md
└── LICENSE
~~~

## Known limitations

- Multi-file or sharded GGUF repositories are not downloaded as a single model; choose an available single-file quantization.
- Speculative decoding uses llama.cpp prompt lookup support and does not automatically download a separate draft model.
- Native GPU acceleration depends on the selected llama-cpp-python wheel and driver/runtime compatibility.
- llama.cpp backends are native build choices. Selecting CUDA or Vulkan validates the installed runtime; it does not transform one native build into another without installation and restart.
- SSD-backed mmap can make an oversized model loadable under memory pressure, but it is not zero-cost layer streaming and can be much slower than a RAM-resident working set.
- The legacy `disk_streamed_layers` API property is retained only as a deprecated layer-equivalent pressure estimate. All non-GPU layers execute on the CPU; `mapped_pressure_layers` and `storage_mode` describe file-backed paging truthfully.
- Managed re-quantization requires a trusted llama-quantize executable and may reduce quality further when the source model is already quantized.
- Generated installers, model files, caches, and frozen sidecar binaries are not committed.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+N | New chat |
| Ctrl+K | Focus model search |
| Ctrl+, | Open settings |
| Ctrl+Shift+O | Open system optimizer |
| Ctrl+Shift+E | Open Extreme Model Center |
| Ctrl+B | Toggle sidebar |
| Ctrl+. | Toggle telemetry panel |
| Enter | Send message |
| Shift+Enter | Insert a new line |
| Esc | Stop generation |

## Contributing

1. Create a focused branch from main.
2. Keep frontend, backend, Rust, and packaging changes scoped and documented.
3. Add or update tests for behavior changes.
4. Run the relevant build, test, audit, and formatting commands locally.
5. Open a pull request with a concise summary, validation results, and screenshots for UI changes.

Please do not commit model weights, API keys, local databases, build directories, or generated installers.

## License

AI Runner is released under the [MIT License](ai-runner/LICENSE).
