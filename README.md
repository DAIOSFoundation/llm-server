# LLM Server (llama.cpp + React)

GGUF 기반 LLM을 **`llama.cpp`의 `llama-server`**로 서빙하고, React UI로 설정/채팅/모니터링하는 프로젝트입니다.

- **클라이언트(UI)**: `frontend/` (React + Vite)
- **서버(추론/라우터)**: `llama.cpp/build/bin/llama-server` (Router Mode 권장)
- **데스크톱(선택 사항)**: Electron 래퍼 (`npm run desktop`)

---

## 구조

```text
llm-server/
├─ frontend/                 # React(Vite) 클라이언트
├─ llama.cpp/                # llama.cpp (Git submodule)
├─ models-config.json        # (서버) 모델 로드 옵션 저장 파일 (router)
├─ user_pw.json              # (서버) 수퍼관리자 비밀번호 해시 저장 파일 (gitignore)
├─ main.js / preload.js      # (선택) Electron 래퍼
└─ package.json              # 실행/빌드 스크립트
```

---

## 주요 기능

- **로그인(수퍼관리자)**
  - 최초 실행 시 수퍼관리자 계정 생성 → 이후 로그인
  - 비밀번호는 서버 파일 `user_pw.json`에 **salt + 반복 SHA-256 해시**로 저장 (평문 저장 없음)
- **모델 설정/관리**
  - 모델 ID(=router에서의 모델 이름) 기반으로 다중 모델을 추가/수정/삭제
  - 설정 저장 시 서버에 모델 로드 설정을 전달하고, 라우터에서 모델을 unload/load 하여 적용
- **GGUF 메타데이터(양자화) 표시**
  - `POST /gguf-info`로 GGUF를 읽고, **양자화/텐서 타입/QKV 타입** 요약을 표시
  - 설정 로드 시 모델 ID가 있으면 **자동으로 메타데이터를 조회**해 요약을 표시
- **실시간 성능 지표(푸시 기반)**
  - `GET /metrics/stream`(SSE)로 VRAM/메모리/CPU/토큰 속도 등을 **폴링 없이** 갱신
- **GPU “사용률(연산 바쁨%)”**
  - 현재 UI의 GPU 게이지는 **실제 GPU 연산 사용률이 아니라 VRAM 점유율(%)**을 표시합니다.
  - `llama.cpp` 기본 metrics에는 플랫폼 공통의 “GPU 바쁨%” 지표가 없어서, 현 시점에서는 **Linux 서비스 환경을 기준으로 별도 구현이 필요**합니다.
  - (향후) Linux에서는 다음 중 한 가지 방식으로 GPU 사용률을 구현하는 것이 현실적입니다.
    - **NVIDIA**: `NVML`(예: `nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetMemoryInfo`) 기반으로 `gpuUtil%/vramUsed/vramTotal` 수집
    - **AMD**: `rocm-smi` / `libdrm` / sysfs 기반 수집
    - **Intel**: `intel_gpu_top`/sysfs 기반 수집
  - 구현 위치는 “서버 측 metrics 수집(server task)”에 통합하여 `/metrics` 및 `/metrics/stream`에 포함시키는 방향을 권장합니다.
- **가이드 페이지**
  - Curl/JS/React/Python/Java/C#/C++ 예제 카드 제공(기본 접힘)

---

## 서버 API 요약

- **Health**: `GET /health`
- **Models (Router Mode)**: `GET /models`
- **Model control (Router Mode)**
  - `POST /models/load` / `POST /models/unload`
  - `GET /models/config` / `POST /models/config` (서버 측 `models-config.json`에 저장)
- **Metrics**
  - `GET /metrics` (Prometheus 텍스트)
  - `GET /metrics/stream` (SSE, 실시간 패널용)
- **GGUF info**: `POST /gguf-info` (서버가 GGUF 파일을 읽고 메타데이터 JSON 반환)
- **UI Auth**
  - `GET /auth/status`
  - `POST /auth/setup`
  - `POST /auth/login`
  - `POST /auth/logout`

---

## 설치

### 공통 요구사항

- **Node.js**: v18+
- **CMake** + C/C++ 컴파일 환경

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

## 빌드

### 1) 의존성 설치

```bash
npm install
npm install --prefix frontend
```

### 2) llama.cpp 빌드 (llama-server)

```bash
cd llama.cpp
mkdir -p build && cd build
cmake ..
cmake --build . --config Release -j 8
cd ../..
```

---

## 실행(개발)

### 서버 실행 (Router Mode)

```bash
npm run server
```

기본 실행 옵션(루트 `package.json`):
- `--port ${LLAMA_PORT:-8080}`
- `--metrics`
- `--models-dir "./llama.cpp/models"`
- `--models-config "./models-config.json"`

### 클라이언트 실행

```bash
npm run client
```

- 기본 접속: `http://localhost:5173/`

### Electron(선택)

```bash
npm run desktop
```

---

## 설정/데이터 저장 위치

### 클라이언트 설정

- 클라이언트(브라우저/Vite) 모드에서는 모델/활성 모델 등을 **localStorage**에 저장합니다.
  - `llmServerClientConfig`
  - `modelConfig` (채팅 요청 파라미터용)

### 서버 설정

- `models-config.json`: 모델 로드 옵션(예: `contextSize`, `gpuLayers`)을 서버가 읽고/저장
- `user_pw.json`: 수퍼관리자 계정 비밀번호 해시 저장 파일 (gitignore)

---

## 배포(패키징)

Electron 패키징(선택):

```bash
npm run build
```

- `frontend`를 프로덕션 빌드하고(`frontend/dist`)
- `electron-builder`로 패키징합니다.
- `llama-server` 바이너리는 `extraResources`로 포함됩니다.

---

## 운영/관리 팁

### 수퍼관리자 초기화(계정 삭제)

테스트 계정을 지우고 다시 생성하려면 서버를 중지한 뒤 `user_pw.json`을 삭제하면 됩니다.

```bash
rm -f ./user_pw.json
npm run server
```

### 환경 변수

- **클라이언트**
  - `VITE_LLAMACPP_BASE_URL`: API 베이스 URL (기본 `http://localhost:8080`)
- **서버**
  - `LLAMA_PORT`: 서버 포트 (기본 8080)

---

## 보안 주의

현재 UI 로그인은 **로컬 개발/단일 사용자 환경**을 전제로 한 경량 구현입니다.
원격/다중 사용자 환경에서 운영할 경우, TLS/정식 인증/세션 저장소/권한 분리 등을 별도로 도입하는 것을 권장합니다.
