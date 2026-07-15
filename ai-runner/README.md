# AI Runner

> A production-oriented local GGUF model runner for Windows, built with React, FastAPI, llama.cpp, and Tauri.

[![CI](https://github.com/Omerfaruk-aydn/Strata/actions/workflows/ci.yml/badge.svg)](https://github.com/Omerfaruk-aydn/Strata/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D4.svg)](https://www.microsoft.com/windows)

AI Runner lets you discover, download, configure, and run GGUF language models locally. It combines a React desktop interface with an authenticated FastAPI service and a Tauri 2 shell. After a model is available locally, inference, prompts, completions, sessions, and settings remain on the machine.

## Highlights

- Local-first inference using llama.cpp and GGUF models.
- Hugging Face search with exact quantization selection.
- Resumable HTTP Range downloads with progress, speed, ETA, cancellation, and partial-file recovery.
- GGUF magic-byte validation, optional SHA-256 verification, safe filenames, and disk/cache limits.
- GPU layer planning, selected GPU support, multi-GPU tensor splitting, CPU thread controls, mmap, mlock, Flash Attention, and KV-cache quantization.
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

Pinned runtime dependencies are in [backend/requirements.txt](backend/requirements.txt). Development and CI dependencies are separated into [backend/requirements-dev.txt](backend/requirements-dev.txt) and [backend/requirements-ci.txt](backend/requirements-ci.txt).

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

- 175 backend tests passed.
- 74.51% backend source coverage.
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
- Generated installers, model files, caches, and frozen sidecar binaries are not committed.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+N | New chat |
| Ctrl+K | Focus model search |
| Ctrl+, | Open settings |
| Ctrl+Shift+O | Open system optimizer |
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

AI Runner is released under the [MIT License](LICENSE).
