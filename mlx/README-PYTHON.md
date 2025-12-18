# MLX Python 기반 서버

이 디렉토리는 MLX 모델을 서빙하기 위한 Python 기반 서버를 포함합니다.

## 개요

C++ MLX API 구현에서 발생하는 세그폴트와 `[Load::eval_gpu] Not implemented` 오류를 해결하기 위해, Python `mlx_lm` 라이브러리를 사용하는 방식으로 전환했습니다.

## 설치

### 1. Python 의존성 설치

```bash
pip3 install -r requirements.txt
```

또는 직접 설치:

```bash
pip3 install mlx-lm mlx
```

### 2. 모델 준비

MLX 모델이 `models/` 디렉토리에 있어야 합니다. 예:

```
mlx/
├── models/
│   └── deepseek-moe-16b-chat-mlx-q4_0/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer.json
├── engine.py
└── server-python.js
```

### 3. 서버 실행

```bash
# 기본 모델 경로 사용
node server-python.js

# 또는 환경 변수로 모델 경로 지정
export MLX_MODEL_PATH="./models/deepseek-moe-16b-chat-mlx-q4_0"
node server-python.js
```

## 아키텍처

### 프로세스 구조

```
Node.js (server-python.js)
    ↓ IPC (stdin/stdout)
Python (engine.py)
    ↓ MLX API
Metal (GPU)
```

### 통신 프로토콜

Node.js와 Python 간의 통신은 JSON 라인 형식으로 이루어집니다:

**Node.js → Python (요청):**
```json
{"prompt": "안녕하세요", "max_tokens": 512, "temperature": 0.7}
```

**Python → Node.js (응답):**
```json
{"status": "loading", "message": "Loading model..."}
{"status": "ready", "message": "Model loaded successfully"}
{"status": "token", "content": "안"}
{"status": "token", "content": "녕"}
{"status": "done"}
```

## API 엔드포인트

서버는 다음 엔드포인트를 제공합니다:

- `GET /health` - 서버 상태 확인
- `GET /metrics` - 서버 메트릭 (준비 상태, 큐 길이 등)
- `POST /chat` - 채팅 요청 (스트리밍 응답)

### POST /chat 예제

```bash
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "안녕하세요",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64
  }'
```

## 장점

### C++ 구현 대비

1. **안정성**: mlx_lm이 내부적으로 최적화되어 있어 세그폴트가 발생하지 않습니다.
2. **자동 처리**: 샤딩, 구조, 가중치 병합을 자동으로 처리합니다.
3. **유지보수성**: 모델 구조 변경 시 `pip install -U mlx-lm`만으로 업데이트 가능합니다.
4. **복잡한 모델 지원**: DeepSeek-MoE 같은 복잡한 모델도 완벽하게 지원합니다.

### 성능

- 실제 연산은 MLX(C++) → Metal(GPU)에서 이루어지므로, Python은 명령만 내리는 역할입니다.
- C++ 직접 구현과 속도 차이가 거의 없습니다.

### 프로세스 격리

- AI 엔진과 웹 서버가 분리되어 있어, AI 엔진에 문제가 생겨도 웹 서버는 죽지 않습니다.
- 자동 재시작 기능이 포함되어 있습니다.

## 문제 해결

### Python을 찾을 수 없음

```bash
# Python 3 설치 확인
python3 --version

# 가상환경 사용 시
source venv/bin/activate
pip install -r requirements.txt
```

### 모델을 찾을 수 없음

```bash
# 모델 경로 확인
ls -la models/

# 환경 변수로 경로 지정
export MLX_MODEL_PATH="./models/your-model-name"
```

### mlx_lm 설치 오류

```bash
# 최신 버전으로 업데이트
pip3 install --upgrade mlx-lm mlx
```

## 레거시 C++ 구현

C++ 기반 서버(`server.js`)는 여전히 사용 가능하지만, 복잡한 모델에서 문제가 발생할 수 있습니다. Python 기반 서버를 사용하는 것을 권장합니다.

