# LLM Server - Electron & llama.cpp 기반 GGUF 모델 서빙 애플리케이션

React, Electron, llama.cpp를 사용하여 GGUF 모델을 로컬에서 직접 서빙하고 상호작용할 수 있는 데스크톱 애플리케이션입니다.

## 주요 기능

-   **다중 모델 관리:** 여러 GGUF 모델 설정을 추가, 편집, 삭제할 수 있습니다.
-   **간편한 모델 선택:** 채팅창 헤더의 드롭다운 메뉴에서 실시간으로 모델을 전환할 수 있습니다.
-   **상세한 추론 설정:** Temperature, Top-K, Top-P 등 다양한 모델 파라미터를 UI에서 직접 제어할 수 있습니다.
-   **파일 기반 설정 저장:** 모든 모델 설정은 JSON 파일에 저장되어 영구적으로 보존됩니다.
-   **다국어 지원:** 한국어와 영어를 지원합니다.
-   **자동 서버 관리:** 앱 시작 시 또는 모델 변경 시 `llama.cpp` 서버가 자동으로 실행/재시작됩니다.

## 프로젝트 구조

```
llm-server/
├── frontend/             # React (Vite) 소스 코드
├── llama.cpp/            # llama.cpp Git 서브모듈 또는 소스 코드
├── main.js               # Electron 메인 프로세스
├── preload.js            # Electron Preload 스크립트
├── package.json          # 프로젝트 실행 및 빌드 스크립트
└── README.md
```

## 설치 및 실행 (개발 환경)

개발 환경에서 앱을 실행하려면 아래 단계를 따르세요.

### 1. 사전 준비

-   **Node.js:** v18 이상 설치
-   **C++ 컴파일러:** macOS의 경우 Xcode Command Line Tools, Windows의 경우 Visual Studio, Linux의 경우 `build-essential` 등이 필요합니다.

### Linux 환경 추가 요구사항

리눅스에서 빌드하려면 `cmake`와 컴파일러가 필요합니다. (Ubuntu/Debian 기준)

```bash
sudo apt update
sudo apt install build-essential cmake
```

### 2. 의존성 설치

프로젝트 루트 디렉터리에서 `frontend`와 루트의 `node_modules`를 모두 설치합니다.

```bash
# 루트 의존성 설치 (Electron, concurrently 등)
npm install

# Frontend 의존성 설치 (React 등)
npm install --prefix frontend
```

### 3. llama.cpp 빌드

`llama.cpp` 서버를 사용하기 위해 먼저 소스 코드를 빌드해야 합니다.

```bash
# llama.cpp 디렉터리로 이동
cd llama.cpp

# build 디렉터리 생성 및 cmake 설정
mkdir -p build && cd build
cmake ..

# 빌드 실행
cmake --build . --config Release

# 프로젝트 루트로 복귀
cd ../..
```
빌드가 성공적으로 완료되면 `llama.cpp/build/bin/` 디렉터리 안에 `llama-server` 실행 파일이 생성됩니다.

### 4. 개발 서버 실행

모든 준비가 완료되었습니다. 이제 다음 명령어로 개발 서버를 시작하세요.

```bash
npm start
```

이 명령어는 React 개발 서버와 Electron 앱을 동시에 실행합니다.

## 설치 패키지 빌드 (배포용)

사용자가 쉽게 설치할 수 있는 `.dmg`(macOS) 또는 `.exe`(Windows) 파일을 만들 수 있습니다.

### 빌드 명령어

프로젝트 루트에서 다음 명령어를 실행하세요.

```bash
npm run build
```

이 명령어는 다음 작업을 자동으로 수행합니다.
1.  React 앱을 프로덕션용으로 빌드합니다 (`frontend/dist`).
2.  `electron-builder`가 `llama-server` 바이너리 파일과 빌드된 React 앱을 포함하여 설치 패키지를 생성합니다.

빌드가 완료되면 프로젝트 루트에 `dist` 폴더가 생성되고, 그 안에 최종 설치 파일이 만들어집니다.

## 설정 파일 (config.json) 안내

이 애플리케이션은 사용자가 생성한 모델 설정과 마지막으로 선택한 모델 정보를 `config.json` 파일에 저장합니다.

### 1. 설정 파일 위치

운영체제별로 설정 파일의 기본 저장 경로는 다음과 같습니다. (Electron의 `userData` 경로)

- **macOS:**
  - `~/Library/Application Support/llm-server/config.json`
  - (개발 모드 실행 시: `~/Library/Application Support/Electron/config.json`)
- **Windows:**
  - `%APPDATA%\llm-server\config.json`
  - (개발 모드 실행 시: `%APPDATA%\Electron\config.json`)
- **Linux:**
  - `~/.config/llm-server/config.json`
  - (개발 모드 실행 시: `~/.config/Electron/config.json`)

### 2. 설정 방법

**방법 A: 애플리케이션 UI 사용 (권장)**
앱 내의 **'설정 (Settings)'** 페이지에서 모델을 추가하고 파라미터를 수정하는 것을 권장합니다. 저장 버튼을 누르면 `config.json` 파일이 자동으로 업데이트됩니다.

**방법 B: config.json 직접 수정**
앱을 종료한 상태에서 `config.json` 파일을 텍스트 에디터로 열어 직접 수정할 수 있습니다.
JSON 형식이 깨지지 않도록 주의해야 합니다.

**주요 설정값 예시:**
- `maxTokens` (n_predict): 한 번에 생성할 최대 토큰 수 (예: 256, 512). 너무 크면 문장이 안 끝나고 횡설수설할 수 있습니다.
- `repeatPenalty`: 반복 억제 강도 (기본: 1.1). 값이 클수록 반복을 강하게 막습니다.
- `dryMultiplier`: DRY(Do Not Repeat Yourself) 샘플링 강도. 0.8 정도로 설정하면 반복을 매우 강력하게 차단합니다.

