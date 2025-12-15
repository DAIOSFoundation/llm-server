# LLM Server (llama.cpp + React)

This project serves GGUF-based LLMs via **`llama.cpp`’s `llama-server`** and provides a React UI for configuration, chat, and monitoring.

- **Client (UI)**: `frontend/` (React + Vite)
- **Server (Inference/Router)**: `llama.cpp/build/bin/llama-server` (Router Mode recommended)
- **Desktop (optional)**: Electron wrapper (`npm run desktop`)

---

## Structure

```text
llm-server/
├─ frontend/                 # React (Vite) client
├─ llama.cpp/                # llama.cpp (Git submodule)
├─ models-config.json        # (server) per-model load options (router)
├─ user_pw.json              # (server) super-admin password hash file (gitignore)
├─ main.js / preload.js      # (optional) Electron wrapper
└─ package.json              # run/build scripts
```

---

## Key Features

- **Login (Super Admin)**
  - Create a super-admin account on first run → then login on subsequent runs
  - Password is stored in `user_pw.json` on the server as **salt + iterative SHA-256 hash** (no plaintext storage)
- **Model Configuration & Management**
  - Add/edit/delete multiple models by **Model ID** (= model name in router mode)
  - When saving settings, the UI sends model load settings to the server and applies them via router unload/load
- **GGUF Metadata (Quantization) Display**
  - Reads GGUF via `POST /gguf-info` and shows a summary of **quantization / tensor types / QKV types**
  - When loading settings, if a model ID exists, metadata is **fetched automatically** and the summary is shown
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

---

## Run (Development)

### Run the server (Router Mode)

```bash
npm run server
```

Default options (root `package.json`):

- `--port ${LLAMA_PORT:-8080}`
- `--metrics`
- `--models-dir "./llama.cpp/models"`
- `--models-config "./models-config.json"`

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

- `models-config.json`: server-side per-model load options (e.g., `contextSize`, `gpuLayers`)
- `user_pw.json`: super-admin password hash file (gitignored)

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

## Operations Tips

### Reset super-admin (delete account)

To remove the test account and re-create it, stop the server and delete `user_pw.json`:

```bash
rm -f ./user_pw.json
npm run server
```

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
