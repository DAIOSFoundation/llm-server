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
├─ auth-server.js            # Authentication server (port 8082)
├─ start-client-server.js    # Client server manager (port 8083)
├─ config.json               # Client-side model configuration
├─ models-config.json        # (server) per-model load options (router)
├─ .auth.json                # Super-admin password hash file (PBKDF2, gitignore)
├─ main.js / preload.js      # (optional) Electron wrapper
└─ package.json              # run/build scripts
```

---

## Key Features

- **Dual Model Format Support (GGUF/MLX)**
  - **GGUF Format**: Uses `llama.cpp`'s `llama-server` for GGUF models (Port 8080)
  - **MLX Format**: Uses MLX C++ API for MLX models (Port 8081, Apple Silicon optimized)
  - **Dual Server Architecture**: Both servers run simultaneously; frontend automatically selects the correct port based on model format
  - No server restart required when switching models - frontend simply changes the endpoint
- **Login (Super Admin)**
  - Separate authentication server (Port 8082) - login works even if model servers are down
  - Create a super-admin account on first run → then login on subsequent runs
  - Password is stored in `.auth.json` as **PBKDF2 hash** (no plaintext storage)
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

## Server Architecture

The application uses a **multi-server architecture** with separate ports for different services:

- **GGUF Server** (Port 8080): `llama.cpp`'s `llama-server` for GGUF models
- **MLX Server** (Port 8081): MLX C++ API server for MLX models
- **Authentication Server** (Port 8082): Handles login/logout/setup independently
- **Client Server Manager** (Port 8083): Manages model servers in client-only mode

Both GGUF and MLX servers run **simultaneously**. The frontend automatically selects the correct port based on the selected model's format.

## Server API Summary

### llama.cpp Server (GGUF Models) - Port 8080

- **Health**: `GET /health`
- **Models (Router Mode)**: `GET /models`
- **Model control (Router Mode)**
  - `POST /models/load` / `POST /models/unload`
  - `GET /models/config` / `POST /models/config` (stored in server-side `models-config.json`)
- **Completion**: `POST /completion` (streaming SSE)
- **Metrics**
  - `GET /metrics` (Prometheus text)
  - `GET /metrics/stream` (SSE, for the real-time panel)
- **GGUF info**: `POST /gguf-info` (server reads a GGUF file and returns metadata JSON)
- **Tokenization**: `POST /tokenize` (text to tokens)
- **Logs**: `GET /logs/stream` (SSE, server logs)

### MLX Server (MLX Models) - Port 8081

- **Health**: `GET /health`
- **Models**: `GET /models`
- **Completion**: `POST /completion` (streaming SSE)
- **Metrics**
  - `GET /metrics` (VRAM and performance metrics)
  - `GET /metrics/stream` (SSE, for the real-time panel)
- **Tokenization**: `POST /tokenize` (text to tokens)
- **Model Verification**: `POST /mlx-verify` (verify MLX model directory and config.json)
- **Logs**: `GET /logs/stream` (SSE, server logs)

### Authentication Server - Port 8082

- **Status**: `GET /auth/status`
- **Setup**: `POST /auth/setup` (create super-admin account)
- **Login**: `POST /auth/login`
- **Logout**: `POST /auth/logout`

### Client Server Manager - Port 8083

- **Save Config**: `POST /api/save-config` (save model configuration, triggers server management)

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

#### Client Mode (Standalone Frontend)

클라이언트 모드에서는 `start-client-server.js`가 자동으로 두 서버를 관리합니다:

```bash
npm run client:all  # 프론트엔드 + 클라이언트 서버 관리자 동시 실행
```

또는 개별 실행:

```bash
npm run client        # 프론트엔드만 실행
npm run client:server # 클라이언트 서버 관리자만 실행 (포트 8083)
```

**클라이언트 서버 관리자**는:
- 초기 로드 시 config.json의 모든 모델을 확인하여 GGUF와 MLX 서버를 모두 시작
- 모델 변경 시 서버를 재시작하지 않고 config만 저장 (프론트엔드가 자동으로 올바른 포트로 요청)
- `config.json` 파일 변경을 감시하여 자동으로 서버 관리

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

- `config.json`: client-side model configuration (active model, model list, settings)
- `models-config.json`: server-side per-model load options (e.g., `contextSize`, `gpuLayers`, `modelFormat`)
- `.auth.json`: super-admin password hash file (PBKDF2, gitignored)

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

To remove the super-admin account and re-create it, delete `.auth.json`:

```bash
rm -f ./.auth.json
# Restart authentication server (or restart the application)
```

The authentication server runs independently on port 8082, so login/setup works even if model servers are down.

### Switching Between Model Formats

**Dual Server Architecture**: Both GGUF and MLX servers run simultaneously on different ports (8080 and 8081).

When you change the model in the dropdown:
1. **No server restart required** - both servers are already running
2. Frontend automatically selects the correct port based on model format:
   - GGUF models → Port 8080
   - MLX models → Port 8081
3. All API calls (health check, chat, metrics) automatically use the correct endpoint
4. `PerformancePanel` automatically reconnects to the correct metrics stream

**Initial Load**: On first load or page refresh, `start-client-server.js` automatically starts both servers if models are configured.

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
