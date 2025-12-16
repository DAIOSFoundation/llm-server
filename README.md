# LLM Server (llama.cpp + MLX + React)

This project serves LLMs via **`llama.cpp`'s `llama-server`** (GGUF format) and **MLX C++ API** (MLX format), providing a React UI for configuration, chat, and monitoring.

- **Client (UI)**: `frontend/` (React + Vite)
- **Server (Inference/Router)**: 
  - `llama.cpp/build/bin/llama-server` (GGUF models, Router Mode recommended)
  - `mlx/server.js` (MLX models, C++ native module)
- **Desktop (optional)**: Electron wrapper (`npm run desktop`)

---

## Structure

```text
llm-server/
├─ frontend/                 # React (Vite) client
├─ llama.cpp/                # llama.cpp (Git submodule)
├─ mlx/                      # MLX server (C++ native module)
│  ├─ src/mlx_server.cpp    # MLX C++ API implementation
│  ├─ server.js              # MLX HTTP server
│  ├─ native.js              # Node.js wrapper
│  └─ models/                # MLX model directory
├─ native/                   # Metal VRAM monitor (native module)
├─ models-config.json        # (server) per-model load options (router)
├─ user_pw.json              # (server) super-admin password hash file (gitignore)
├─ main.js / preload.js      # (optional) Electron wrapper
└─ package.json              # run/build scripts
```

---

## Key Features

- **Dual Model Format Support**
  - **GGUF Format**: Uses `llama.cpp`'s `llama-server` for GGUF models
  - **MLX Format**: Uses MLX C++ API for MLX models (Apple Silicon optimized)
  - Automatic server switching based on selected model format
- **Login (Super Admin)**
  - Create a super-admin account on first run → then login on subsequent runs
  - Password is stored in `user_pw.json` on the server as **salt + iterative SHA-256 hash** (no plaintext storage)
- **Model Configuration & Management**
  - Add/edit/delete multiple models by **Model ID** (= model name in router mode)
  - Model format selection (GGUF/MLX) with automatic path validation
  - When saving settings, the UI sends model load settings to the server and applies them via router unload/load
- **GGUF Metadata (Quantization) Display**
  - Reads GGUF via `POST /gguf-info` and shows a summary of **quantization / tensor types / QKV types**
  - When loading settings, if a model ID exists, metadata is **fetched automatically** and the summary is shown
- **MLX C++ Native Implementation**
  - Direct MLX C++ API integration (no Python subprocess)
  - Full Transformer forward pass implementation (Multi-Head Attention, Feed Forward, Layer Normalization)
  - BPE tokenization/detokenization (llama.cpp style)
  - Advanced sampling strategies (Temperature, Top-K, Top-P, Min-P)
  - Streaming token generation with async callbacks
- **Real-time Performance Metrics (Push-based)**
  - Updates VRAM / memory / CPU / token speed via `GET /metrics/stream` (SSE) **without polling**
- **GPU “Utilization (Compute Busy %)”**
  - The current GPU gauge in the UI shows **VRAM occupancy (%)**, not actual GPU compute utilization.
  - `llama.cpp`’s default metrics do not expose a cross-platform “GPU busy %”, so **additional implementation is required for Linux production**.
  - (Future) On Linux, these approaches are practical:
    - **NVIDIA**: NVML (e.g., `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetMemoryInfo`) to collect `gpuUtil%/vramUsed/vramTotal`
    - **AMD**: `rocm-smi` / `libdrm` / sysfs-based collection
    - **Intel**: `intel_gpu_top` / sysfs-based collection
  - Recommended implementation location: integrate into server-side metrics collection (server task) and expose via `/metrics` and `/metrics/stream`.
- **Guide Page**
  - Provides example cards for Curl / JS / React / Python / Java / C# / C++ (collapsed by default)

---

## Server API Summary

### llama.cpp Server (GGUF Models)

- **Health**: `GET /health`
- **Models (Router Mode)**: `GET /models`
- **Model control (Router Mode)**
  - `POST /models/load` / `POST /models/unload`
  - `GET /models/config` / `POST /models/config` (stored in server-side `models-config.json`)
- **Metrics**
  - `GET /metrics` (Prometheus text)
  - `GET /metrics/stream` (SSE, for the real-time panel)
- **GGUF info**: `POST /gguf-info` (server reads a GGUF file and returns metadata JSON)
- **UI Auth**
  - `GET /auth/status`
  - `POST /auth/setup`
  - `POST /auth/login`
  - `POST /auth/logout`

### MLX Server (MLX Models)

- **Health**: `GET /health`
- **Completion**: `POST /completion` (streaming SSE)
- **Metrics**
  - `GET /metrics` (VRAM and performance metrics)
  - `GET /metrics/stream` (SSE, for the real-time panel)
- **Tokenization**: `POST /tokenize` (text to tokens)
- **Model Verification**: `POST /mlx-verify` (verify MLX model directory and config.json)

---

## Installation

### Common Requirements

- **Node.js**: v18+
- **CMake** + a working C/C++ toolchain

### macOS

- Xcode Command Line Tools:

```bash
xcode-select --install
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y build-essential cmake
```

---

## Build

### 1) Install dependencies

```bash
npm install
npm install --prefix frontend
```

### 2) Build llama.cpp (llama-server)

```bash
cd llama.cpp
mkdir -p build && cd build
cmake ..
cmake --build . --config Release -j 8
cd ../..
```

### 3) Build MLX C++ Library (for MLX models)

MLX C++ 라이브러리를 설치해야 합니다:

```bash
# MLX C++ 라이브러리 소스에서 빌드 및 설치
git clone https://github.com/ml-explore/mlx.git /tmp/mlx-build
cd /tmp/mlx-build
mkdir build && cd build
cmake ..
make -j8
# 헤더 파일과 라이브러리 수동 설치
sudo mkdir -p /opt/homebrew/include/mlx
sudo mkdir -p /opt/homebrew/lib
sudo cp -r ../mlx/*.h /opt/homebrew/include/mlx/
sudo cp -r ../mlx/io/*.h /opt/homebrew/include/mlx/io/
sudo cp -r ../mlx/backend/*.h /opt/homebrew/include/mlx/backend/
sudo cp build/libmlx*.dylib /opt/homebrew/lib/
```

### 4) Build Native Modules

```bash
npm run build:native
```

이 명령은 다음을 빌드합니다:
- `native/`: Metal VRAM monitor (macOS)
- `mlx/`: MLX C++ native module

---

## Run (Development)

### Run the server

#### GGUF Models (llama.cpp)

```bash
npm run server
```

Default options (root `package.json`):

- `--port ${LLAMA_PORT:-8080}`
- `--metrics`
- `--models-dir "./llama.cpp/models"`
- `--models-config "./models-config.json"`

#### MLX Models

MLX 서버는 자동으로 시작됩니다. 모델 형식을 MLX로 선택하면 자동으로 MLX 서버가 시작되고 llama.cpp 서버는 중지됩니다.

또는 수동으로 실행:

```bash
npm run server:mlx-proxy  # MLX 모델 검증 프록시 (포트 8081)
```

#### Both Servers

```bash
npm run server:all  # llama.cpp 서버 + MLX 프록시 동시 실행
```

### Run the client

```bash
npm run client
```

- Default URL: `http://localhost:5173/`

### Electron (optional)

```bash
npm run desktop
```

---

## Configuration / Data Locations

### Client configuration

- In client-only mode (browser/Vite), model list and active model are stored in **localStorage**:
  - `llmServerClientConfig`
  - `modelConfig` (chat request parameters)

### Server configuration

- `models-config.json`: server-side per-model load options (e.g., `contextSize`, `gpuLayers`, `modelFormat`)
- `user_pw.json`: super-admin password hash file (gitignored)

### Model Directories

- **GGUF Models**: `llama.cpp/models/` (Model ID = directory name)
- **MLX Models**: `mlx/models/` (Model ID = directory name, must contain `config.json` and model weights)

---

## Deployment (Packaging)

Electron packaging (optional):

```bash
npm run build
```

- Builds the `frontend` production bundle (`frontend/dist`)
- Packages via `electron-builder`
- Includes the `llama-server` binary via `extraResources`

---

## MLX Implementation Details

### MLX C++ API Integration

MLX 서버는 Apple의 MLX C++ 공식 라이브러리를 직접 사용합니다:

- **모델 로딩**: `mlx::core::load_safetensors()` 또는 `mlx::core::load_gguf()` 사용
- **Transformer 구현**: llama.cpp의 ggml-metal 구현을 참고하여 MLX C++ API로 구현
  - Multi-Head Attention
  - Feed Forward Network
  - Layer Normalization
- **토큰화**: llama.cpp의 BPE 토큰화 알고리즘을 참고하여 구현
- **샘플링**: Temperature, Top-K, Top-P, Min-P 지원
- **스트리밍**: 비동기 토큰 생성 및 SSE 스트리밍

### MLX Model Requirements

MLX 모델 디렉토리는 다음을 포함해야 합니다:

- `config.json`: 모델 설정 파일
- `model.safetensors` 또는 `*.gguf`: 모델 가중치 파일
- `tokenizer.json`: 토큰화 설정 (선택사항)

### MLX vs GGUF

| Feature | GGUF (llama.cpp) | MLX |
|---------|------------------|-----|
| Platform | Cross-platform | macOS (Apple Silicon) |
| GPU Acceleration | CUDA/Metal/OpenCL | Metal (optimized) |
| Model Format | GGUF | Safetensors/GGUF |
| Tokenization | Built-in | BPE (llama.cpp style) |
| Performance | High | Very High (Apple Silicon) |

## Operations Tips

### Reset super-admin (delete account)

To remove the test account and re-create it, stop the server and delete `user_pw.json`:

```bash
rm -f ./user_pw.json
npm run server
```

### Switching Between Model Formats

When you change the model format (GGUF ↔ MLX) in the settings page, the server automatically:
1. Stops the current server (llama.cpp or MLX)
2. Starts the appropriate server for the selected format
3. Loads the model from the correct directory (`llama.cpp/models/` or `mlx/models/`)

### Environment variables

- **Client**
  - `VITE_LLAMACPP_BASE_URL`: API base URL (default `http://localhost:8080`)
- **Server**
  - `LLAMA_PORT`: server port (default 8080)

---

## Security Notes

The current UI login is a lightweight implementation intended for **local development / single-user** usage.
For remote or multi-user production environments, you should additionally introduce TLS, proper authentication, durable session storage, and authorization controls.

---

### (Future Work) Security Hardening Checklist

- **Transport security (TLS)**
  - Expose externally via HTTPS only (reverse proxy/Ingress); consider mTLS for internal traffic
- **Authentication & session hardening**
  - Strong random session tokens (e.g., 256-bit) + expiry/refresh (sliding/absolute) + server-side revocation on logout
  - Limit concurrent sessions; add delay/blocking on repeated auth failures (brute-force defense); audit logging
  - For browser deployment: consider cookie-based sessions (`HttpOnly`/`Secure`/`SameSite`) and CSRF protection
- **Authorization (RBAC) separation**
  - Apply role-based policies for sensitive endpoints such as model management (`/models/*`, `/models/config`), log streaming (`/logs/stream`), and system metrics (`/metrics*`)
- **Password storage upgrade**
  - Replace custom hashing with a standard KDF (recommended: Argon2id; alternatives: bcrypt/scrypt) + parameter upgrade strategy
- **Secrets & key management**
  - Harden `user_pw.json` permissions (e.g., 600) and storage path; define backup/recovery procedures
  - Never log tokens/passwords/API keys; mask secrets in logs/errors
- **Rate limiting & resource limits**
  - Apply per-IP/per-account rate limiting to `/completion` and streaming endpoints (`/metrics/stream`, `/logs/stream`)
  - Limit request body size; cap concurrent streams/requests (DoS mitigation)
- **Input validation & path safety**
  - Restrict model paths/IDs to an allowlisted directory; prevent `..` and symlink escape
  - Apply file size limits and timeouts when reading/parsing GGUF metadata
- **Operational hardening**
  - Default bind to `127.0.0.1`; expose externally only via a controlled proxy layer
  - Run as least-privileged user; use container isolation (AppArmor/SELinux); operate security updates and SCA scanning
