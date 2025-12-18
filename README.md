# LLM Server (llama.cpp + MLX + React)

This project serves LLMs via **`llama.cpp`'s `llama-server`** (GGUF format) and **MLX Python API** (MLX format), providing a React UI for configuration, chat, and monitoring.

- **Client (UI)**: `frontend/` (React + Vite)
- **Server (Inference/Router)**: 
  - `llama.cpp/build/bin/llama-server` (GGUF models, Router Mode recommended)
  - `mlx/server-python.js` (MLX models, Python mlx_lm library)
- **Desktop (optional)**: Electron wrapper (`npm run desktop`)

---

## Project Structure

### Architecture Overview

This project consists of **4 independent servers** and a **frontend client**:

1. **GGUF Server** (Port 8080): Uses `llama.cpp`'s `llama-server` to serve GGUF format models
2. **MLX Server** (Port 8081): Uses MLX Python API (mlx_lm) to serve MLX format models
3. **Authentication Server** (Port 8082): Handles login/logout/setup independently (runs independently from model servers)
4. **Client Server Manager** (Port 8083): Automatically manages model servers in client-only mode
5. **Frontend** (Port 5173): React + Vite based web UI

### Directory Structure

```text
llm-server/
├─ frontend/                      # React (Vite) client
│  ├─ src/
│  │  ├─ components/              # React components
│  │  ├─ pages/                   # Page components
│  │  ├─ contexts/                # React Context (Auth, etc.)
│  │  └─ services/                # API service layer
│  └─ package.json
│
├─ llama.cpp/                     # llama.cpp (Git submodule)
│  └─ build/bin/llama-server      # GGUF server executable
│
├─ mlx/                           # MLX server (Python 기반)
│  ├─ server-python.js            # MLX HTTP server (port 8081)
│  ├─ engine.py                   # Python mlx_lm 엔진
│  ├─ requirements.txt            # Python 의존성
│  └─ models/                      # MLX model directory
│
├─ native/                        # Metal VRAM monitoring native module
│  ├─ src/
│  │  └─ vram_monitor.cpp         # macOS Metal VRAM monitoring
│  └─ binding.gyp                 # node-gyp build configuration
│
├─ auth-server.js                 # Authentication server (port 8082)
├─ start-client-server.js          # Client server manager (port 8083)
├─ mlx-verify-proxy.js            # MLX model verification proxy (port 8084)
│
├─ config.json                     # Client model configuration (localStorage sync)
├─ models-config.json              # Server-side model load options (router mode)
├─ .auth.json                      # Super admin password hash (PBKDF2, gitignore)
│
├─ main.js / preload.js            # (Optional) Electron wrapper
└─ package.json                    # Run/build scripts
```

### Server Components

#### 1. GGUF Server (Port 8080)
- **Executable**: `llama.cpp/build/bin/llama-server`
- **Native Library**: `llama.cpp` (C++ implementation)
- **Functionality**: GGUF format model loading and inference
- **Startup**: Auto-started by `start-client-server.js` or manually executed

#### 2. MLX Server (Port 8081)
- **Server File**: `mlx/server-python.js` (Python 기반)
- **Python Engine**: `mlx/engine.py` (uses mlx_lm library)
- **Functionality**: MLX format model loading and inference
- **Startup**: Auto-started by `start-client-server.js` or manually executed
- **Note**: Python 기반 서버는 mlx_lm 라이브러리를 사용하여 안정적으로 모델을 로드하고 추론을 수행합니다.

#### 3. Authentication Server (Port 8082)
- **Server File**: `auth-server.js`
- **Functionality**: User authentication (login/logout/setup), super admin account management
- **Independence**: Runs independently from model servers (login works even if model servers are down)

#### 4. Client Server Manager (Port 8083)
- **Server File**: `start-client-server.js`
- **Role**: Central manager that automatically manages model servers in client-only mode (running frontend only in browser)
- **Key Features**:
  1. **Automatic Server Startup/Management**
     - Reads `config.json` on initial load and automatically starts both GGUF and MLX servers
     - Monitors `config.json` file changes (fs.watchFile)
     - Starts appropriate server based on model format (GGUF → port 8080, MLX → port 8081)
  
  2. **Frontend Configuration Synchronization**
     - Handles model configuration save requests from frontend via `/api/save-config` endpoint
     - Saves received configuration to `config.json` file
     - Automatically starts servers if needed after saving configuration
  
  3. **Server Process Management**
     - GGUF Server: Spawns and manages `llama.cpp/build/bin/llama-server` process
     - MLX Server: Spawns and manages `mlx/server-python.js` Node.js process
     - Handles server process termination and restart
  
  4. **Client Mode Support**
     - Required when running frontend only in browser without Electron
     - Frontend cannot directly start servers, so a separate Node.js process manages servers
     - Acts as a bridge between frontend and servers
  
- **Use Cases**:
  - Development: Auto-starts with frontend when running `npm run client:all`
  - Production: Run as a separate process to allow frontend to control servers

#### 5. Frontend (Port 5173)
- **Framework**: React + Vite
- **Functionality**: 
  - Model configuration and management UI
  - Chat interface
  - Real-time performance monitoring
  - Server log streaming

### Native Modules

#### llama.cpp
- **Location**: `llama.cpp/` (Git submodule)
- **Purpose**: GGUF model loading and inference
- **Build**: Uses CMake to generate `llama-server` executable

#### MLX Python Library
- **Installation**: `pip install mlx-lm mlx`
- **Purpose**: MLX model loading and inference (Apple Silicon optimized)
- **Integration**: Used via Python subprocess in `mlx/engine.py`

#### Native VRAM Monitor
- **Location**: `native/src/vram_monitor.cpp`
- **Purpose**: macOS Metal VRAM usage monitoring
- **Build**: Uses `node-gyp` to build as Node.js native module

---

## Key Features

- **Dual Model Format Support (GGUF/MLX)**
  - **GGUF Format**: Uses `llama.cpp`'s `llama-server` for GGUF models (Port 8080)
  - **MLX Format**: Uses MLX Python API (mlx_lm) for MLX models (Port 8081, Apple Silicon optimized)
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
- **MLX Python Implementation**
  - Python mlx_lm library integration via subprocess
  - Automatic model loading, sharding, and weight merging
  - Built-in tokenization support
  - Sampling strategies (Temperature, Top-P, Repetition Penalty)
  - Streaming token generation with SSE
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

### Default Ports

- **GGUF Server**: Port **8080** (default)
  - Uses `llama.cpp`'s `llama-server` for GGUF models
  - Started via `start-client-server.js` with `--port 8080` flag
- **MLX Server**: Port **8081** (default)
  - Uses MLX Python API (mlx_lm) server for MLX models
  - Configured in `mlx/server-python.js` (default: 8081)
- **Authentication Server**: Port **8082** (default)
  - Handles login/logout/setup independently
  - Runs even when model servers are down
- **Client Server Manager**: Port **8083** (default)
  - Manages model servers in client-only mode
  - Receives config updates from frontend via `/api/save-config`

### Port Configuration

- **GGUF Server Port**: Can be changed via `LLAMA_PORT` environment variable (default: 8080)
- **MLX Server Port**: Hardcoded in `mlx/server-python.js` (default: 8081)
- **Authentication Server Port**: Hardcoded in `auth-server.js` (default: 8082)
- **Client Server Manager Port**: Hardcoded in `start-client-server.js` (default: 8083)

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

- **Save Config**: `POST /api/save-config`
  - Model configuration save request from frontend
  - Request body: `{ models: [...], activeModelId: "..." }`
  - Action: Saves to `config.json` file and automatically starts servers if needed
  - Response: `{ success: true }`

**Client Server Manager Role**:
- Automatically manages model servers in client-only mode (running frontend only in browser)
- Acts as a bridge between frontend and servers, allowing frontend to control servers without Electron
- Monitors `config.json` file and automatically starts/manages servers when model configuration changes

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

### 3) MLX 서버 설정 (MLX 모델 사용 시)

MLX 서버는 Python 기반으로 동작합니다.

```bash
# Python 의존성 설치
cd mlx
pip3 install -r requirements.txt
# 또는 직접 설치
pip3 install mlx-lm mlx

# 서버 실행
node server-python.js
```

**장점:**
- ✅ mlx_lm이 샤딩, 구조, 가중치 병합을 자동으로 처리
- ✅ 모델 구조 변경 시 `pip install -U mlx-lm`만으로 업데이트 가능
- ✅ 안정성과 유지보수성 향상
- ✅ DeepSeek-MoE 같은 복잡한 모델도 완벽하게 지원

**환경 변수:**
```bash
# 모델 경로 지정 (선택사항)
export MLX_MODEL_PATH="./models/deepseek-moe-16b-chat-mlx-q4_0"
node server-python.js
```

### 4) Build Native Modules

```bash
npm run build:native
```

This command builds:
- `native/`: Metal VRAM monitor (macOS)

---

## Run (Development)

### Quick Start

The easiest way to run the entire project:

```bash
npm run client:all
```

This single command starts all required services:
- **Client Server Manager** (port 8083): Automatically manages GGUF and MLX model servers
- **Auth Server** (port 8082): Handles authentication (login/logout/setup)
- **Frontend** (port 5173): React web UI

After starting, open your browser and navigate to: **http://localhost:5173**

The Client Server Manager will automatically start the GGUF server (port 8080) and MLX server (port 8081) based on your model configuration.

### Detailed Run Options

#### Client Mode (Recommended - Standalone Frontend)

In client mode, `start-client-server.js` automatically manages both model servers (GGUF and MLX):

```bash
npm run client:all  # Run all services: frontend + client server manager + auth server
```

**What gets started:**
- **Client Server Manager** (port 8083): Manages GGUF and MLX model servers
- **Auth Server** (port 8082): Handles authentication (login/logout/setup)
- **Frontend** (port 5173): React web UI

**Individual service commands:**

```bash
npm run client        # Run frontend only (port 5173)
npm run client:server # Run client server manager only (port 8083)
npm run client:auth   # Run authentication server only (port 8082)
```

**Client Server Manager behavior:**
- On initial load, checks all models in `config.json` and starts both GGUF and MLX servers
- On model change, saves config only without restarting servers (frontend automatically requests correct port)
- Monitors `config.json` file changes and automatically manages servers

**Note**: When running `npm run client:all`, all three services (Client Server Manager, Auth Server, and Frontend) start together. The Auth Server is required for login functionality.

#### GGUF Server Only (Manual)

To run only the GGUF server manually:

```bash
npm run server
```

Default options (root `package.json`):

- `--port ${LLAMA_PORT:-8080}`
- `--metrics`
- `--models-dir "./llama.cpp/models"`
- `--models-config "./models-config.json"`

#### Electron Mode (Optional)

For desktop application mode:

```bash
npm run desktop
```

This runs the Electron wrapper with the frontend bundled.

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

### MLX Python API Integration

The MLX server uses Python's mlx_lm library:

- **Model Loading**: Uses `mlx_lm.load()` which automatically handles model loading, sharding, and weight merging
- **Transformer Implementation**: Handled by mlx_lm library
  - Multi-Head Attention
  - Feed Forward Network
  - Layer Normalization
- **Tokenization**: Uses mlx_lm's built-in tokenizer
- **Sampling**: Supports Temperature, Top-P, Repetition Penalty
- **Streaming**: Async token generation and SSE streaming via `stream_generate()`

### MLX Model Requirements

MLX model directory must contain:

- `config.json`: Model configuration file
- `model.safetensors` or `*.gguf`: Model weight files
- `tokenizer.json`: Tokenization configuration (optional)

### MLX vs GGUF

| Feature | GGUF (llama.cpp) | MLX |
|---------|------------------|-----|
| Platform | Cross-platform | macOS (Apple Silicon) |
| GPU Acceleration | CUDA/Metal/OpenCL | Metal (optimized) |
| Model Format | GGUF | Safetensors (via mlx_lm) |
| Tokenization | Built-in | mlx_lm built-in tokenizer |
| Performance | High | Very High (Apple Silicon) |
| Implementation | C++ | Python (mlx_lm) |

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

**Client Server Manager Workflow**:

1. **Frontend Start** → Browser accesses `http://localhost:5173`
2. **Config Load** → Frontend loads config from `localStorage` or calls `/api/save-config`
3. **Automatic Server Start** → Client Server Manager reads `config.json` and starts required servers
   - If GGUF model exists → Start GGUF server on port 8080
   - If MLX model exists → Start MLX server on port 8081
4. **Model Switch** → Frontend calls `/api/save-config` when model is selected
5. **Automatic Port Selection** → Frontend automatically requests API to correct port based on selected model format
   - GGUF model selected → Use `http://localhost:8080`
   - MLX model selected → Use `http://localhost:8081`

### Environment variables

- **Client**
  - `VITE_LLAMACPP_BASE_URL`: API base URL (default `http://localhost:8080`)
- **Server**
  - `LLAMA_PORT`: server port (default 8080)

---

## Testing

### MLX Server Testing

MLX 서버의 추론 기능을 테스트하기 위한 자동화된 테스트 스크립트가 제공됩니다.

#### 서버 실행 방법

MLX 서버를 수동으로 실행하려면:

```bash
cd mlx
node server.js
```

서버는 기본적으로 **포트 8081**에서 실행됩니다.

#### 추론 테스트 실행 방법

추론 테스트를 실행하려면:

```bash
cd mlx
node test-inference.js
```

**테스트 스크립트 동작 방식:**

1. **자동 서버 시작**: 테스트 스크립트는 MLX 서버가 실행 중이 아니면 자동으로 시작합니다.
2. **헬스 체크**: 서버가 준비될 때까지 최대 3회 재시도하며, 각 재시도 간 1초 대기합니다.
3. **메트릭 수집**: 서버의 성능 메트릭을 수집합니다.
4. **추론 요청**: 테스트 프롬프트(`안녕, 대한민국의 수도는 어디지?`)로 추론을 요청합니다.
5. **결과 확인**: 추론 결과를 확인하고 성공/실패를 기록합니다.

**테스트 설정:**

- **서버 포트**: 8081 (기본값)
- **테스트 프롬프트**: `안녕, 대한민국의 수도는 어디지?`
- **헬스 체크 재시도**: 최대 3회
- **서버 시작 타임아웃**: 30초
- **추론 타임아웃**: 60초

**테스트 결과:**

테스트가 완료되면 다음 정보가 출력됩니다:
- 성공한 테스트 수
- 실패한 테스트 수
- 각 테스트의 상세 로그

**주의사항:**

- 테스트를 실행하기 전에 MLX 모델이 `mlx/models/` 디렉토리에 올바르게 설정되어 있어야 합니다.
- 서버가 이미 실행 중인 경우, 테스트 스크립트는 기존 서버를 사용합니다.
- 테스트 중 서버가 크래시되면 자동으로 재시작을 시도합니다.

### GGUF Server Testing

GGUF 서버의 테스트는 `llama.cpp`의 기본 테스트 도구를 사용할 수 있습니다:

```bash
cd llama.cpp
# llama-server가 실행 중인 상태에서
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world", "n_predict": 10}'
```

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
