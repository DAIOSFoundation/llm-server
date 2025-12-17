# DeepSeek-MoE-16b MLX 완전 설정 가이드

## 🎯 목표
C++ 추론 엔진에서 사용할 수 있는 **검증되고 통합된 깨끗한 모델 파일** 확보

## 📋 사전 준비

### 하드웨어 요구사항
- Apple Silicon (M1/M2/M3/M4) Mac
- RAM: 16GB 이상 권장
- VRAM: 4-bit 양자화 시 약 9~10GB 소요

### 소프트웨어 요구사항
- Python 3.9 이상
- pip (Python 패키지 관리자)

## 🚀 1단계: 모델 변환 (자동화 스크립트)

### 실행 방법

```bash
cd /Volumes/Transcend/Projects/llm-server/mlx
chmod +x setup_deepseek_mlx.sh
./setup_deepseek_mlx.sh
```

### 스크립트가 수행하는 작업

1. **가상환경 생성**: Python 가상환경 생성 및 활성화
2. **패키지 설치**: mlx-lm, huggingface_hub 설치
3. **모델 다운로드**: Hugging Face에서 DeepSeek-MoE-16b-chat 자동 다운로드
4. **4-bit 양자화**: MLX 형식으로 변환 및 양자화
5. **테스트 실행**: 변환된 모델로 간단한 추론 테스트

### 예상 소요 시간
- 모델 다운로드: 네트워크 속도에 따라 다름 (수십 GB)
- 변환 및 양자화: 약 30분 ~ 1시간 (하드웨어 성능에 따라 다름)

## 🧪 2단계: Python으로 모델 테스트

### 실행 방법

```bash
cd /Volumes/Transcend/Projects/llm-server/mlx
source venv/bin/activate  # 가상환경 활성화
python chat_deepseek.py
```

### 예상 출력

```
Loading model from ./deepseek-16b-mlx-q4...
Model loaded. Start chatting! (Type 'quit' to exit)
--------------------------------------------------
User: 안녕하세요
Assistant: 안녕하세요! 무엇을 도와드릴까요?
```

## 🔧 3단계: C++ 프로젝트에 모델 연결

### 모델 파일 복사

변환된 모델을 C++ 프로젝트의 모델 디렉토리로 복사:

```bash
# 변환된 모델 확인
ls -lh ./deepseek-16b-mlx-q4/

# C++ 프로젝트 모델 디렉토리로 복사
cp -r ./deepseek-16b-mlx-q4/* ./models/deepseek-moe-16b-chat-mlx-q4_0/
```

### 필수 파일 확인

다음 파일들이 있어야 합니다:

```
models/deepseek-moe-16b-chat-mlx-q4_0/
├── config.json              # 모델 설정
├── tokenizer.json           # 토크나이저 설정
├── tokenizer_config.json    # 토크나이저 추가 설정
├── weights.safetensors      # 통합된 가중치 파일 (또는 여러 파일)
└── model.safetensors.index.json  # (여러 파일인 경우)
```

### 가중치 검증

```python
import mlx.core as mx
import json

# Config 확인
with open("models/deepseek-moe-16b-chat-mlx-q4_0/config.json") as f:
    config = json.load(f)

expected_dim = config["hidden_size"]  # 2048

# Weight 확인
weights = mx.load("models/deepseek-moe-16b-chat-mlx-q4_0/weights.safetensors")
q_proj = weights["model.layers.0.self_attn.q_proj.weight"]

print(f"Expected: ({expected_dim}, {expected_dim})")
print(f"Actual: {q_proj.shape}")

if q_proj.shape == (expected_dim, expected_dim):
    print("✅ 모델 파일 정상! C++ 엔진에서 사용 가능합니다.")
else:
    print("❌ 모델 파일 문제 있음 - 재변환 필요")
```

## ✅ 4단계: C++ 서버 테스트

모델 파일이 준비되면 C++ 서버를 실행:

```bash
cd /Volumes/Transcend/Projects/llm-server/mlx
npm run build
node test-server-temp.js
```

### 성공 시나리오

```
[MLX] LoadSafetensors: Loading weights...
[MLX] q_proj shape: (2048, 2048) - Verified ✅
[MLX] o_proj shape: (2048, 2048) - Verified ✅
[MLX] No MLP weights detected in Attention block ✅
[MLX] Server started on port 8081
```

## 🔍 문제 해결

### 문제: "q_proj shape mismatch" 에러

**원인**: 모델 파일이 불완전하거나 샤딩된 상태

**해결**:
1. `setup_deepseek_mlx.sh` 스크립트로 재변환
2. `weights.safetensors` 파일 크기 확인 (약 8-10GB여야 함)
3. 위의 검증 스크립트로 shape 확인

### 문제: "MLP weight detected in Attention" 에러

**원인**: 가중치 키 매핑 오류

**해결**: C++ 코드의 엄격한 검증 로직이 이미 구현되어 있음. 모델 파일이 정상이면 발생하지 않음.

### 문제: 변환 시간이 너무 오래 걸림

**원인**: 하드웨어 성능 또는 네트워크 속도

**해결**:
- 네트워크가 느리면 사전에 모델을 다운로드
- 변환은 한 번만 수행하면 되므로 기다림

## 📊 파일 크기 가이드

정상적인 모델 파일 크기:

- **16B 모델 4-bit 양자화**: 약 8-10GB
- **단일 weights.safetensors**: 8-10GB (통합된 경우)
- **여러 파일로 분할**: 각 파일이 2-4GB (정상)

⚠️ **경고**: 파일이 1-2GB라면 불완전한 파일입니다.

## 🎉 완료 체크리스트

- [ ] `setup_deepseek_mlx.sh` 실행 완료
- [ ] `chat_deepseek.py` 테스트 성공
- [ ] 가중치 검증 스크립트 통과 (q_proj: 2048x2048)
- [ ] C++ 서버 정상 시작
- [ ] 추론 테스트 성공

## 💡 추가 팁

### 가상환경 재사용

```bash
# 다음에 사용할 때
cd /Volumes/Transcend/Projects/llm-server/mlx
source venv/bin/activate
python chat_deepseek.py
```

### 모델 경로 변경

`chat_deepseek.py`의 `model_path` 변수를 수정하여 다른 모델 사용 가능

### 배치 변환

여러 모델을 한 번에 변환하려면 스크립트를 수정하여 반복 실행

---

**작성일**: 2024
**상태**: ✅ 완전한 설정 가이드

