# MLX Server Native Module

MLX 서버를 위한 C++ native 모듈입니다. MLX C++ 공식 라이브러리를 직접 사용합니다.

## 요구사항

- Node.js 14+
- node-gyp
- MLX C++ 라이브러리 (설치 필요)
- macOS (Metal 지원)

## MLX C++ 라이브러리 설치

MLX C++ 라이브러리를 설치해야 합니다. 일반적인 설치 방법:

```bash
# Homebrew를 사용한 설치 (예시)
brew install mlx

# 또는 소스에서 빌드
git clone https://github.com/ml-explore/mlx.git
cd mlx
mkdir build && cd build
cmake ..
make install
```

설치 후 헤더 파일과 라이브러리 경로를 확인하세요:
- 헤더 파일: `/opt/homebrew/include/mlx` 또는 `/usr/local/include/mlx`
- 라이브러리: `/opt/homebrew/lib` 또는 `/usr/local/lib`

## 빌드 방법

```bash
cd mlx
npm install
npm run build
```

또는 프로젝트 루트에서:

```bash
npm run build:native
```

## 빌드 설정 수정

MLX C++ 라이브러리가 다른 경로에 설치된 경우, `binding.gyp` 파일을 수정하세요:

```json
{
  "include_dirs": [
    "<!@(node -p \"require('node-addon-api').include\")",
    "/your/mlx/include/path"
  ],
  "libraries": [
    "-L/your/mlx/lib/path",
    "-lmlx",
    "-lmlx_core"
  ]
}
```

## 사용 방법

이 모듈은 `mlx/server.js`에서 자동으로 로드됩니다. C++ 모듈이 사용 가능하면 자동으로 사용하고, 없으면 Python fallback을 사용합니다.

## 구조

- `src/mlx_server.cpp`: MLX C++ API를 사용한 구현
- `native.js`: Node.js 래퍼
- `server.js`: MLX HTTP 서버
- `binding.gyp`: 빌드 설정
- `models/`: MLX 모델 디렉토리

## MLX C++ API 구현

현재 코드는 MLX C++ API의 전방 선언을 사용하고 있습니다. 실제 MLX C++ 라이브러리를 설치한 후:

1. `src/mlx_server.cpp`의 헤더 파일 include 경로를 실제 MLX C++ 라이브러리 경로로 수정
2. 전방 선언을 실제 헤더 파일 include로 교체
3. 주석 처리된 실제 API 호출 코드를 활성화하고 MLX C++ API 문서에 맞게 수정

## 참고

- MLX C++ API 문서: https://ml-explore.github.io/mlx/
- MLX GitHub: https://github.com/ml-explore/mlx
