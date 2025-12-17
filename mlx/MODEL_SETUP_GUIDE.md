# 모델 설정 가이드

> **⚠️ 중요**: 이 가이드는 간단한 요약입니다. 완전한 설정 가이드는 `COMPLETE_SETUP_GUIDE.md`를 참조하세요.

## 현재 상황
현재 사용 중인 모델 파일이 불완전합니다:
- `q_proj` shape: `(2048, 256)` - 7/8 데이터 누락
- 예상 shape: `(2048, 2048)`

## 빠른 해결 방법 (자동화 스크립트)

### 방법 1: 자동화 스크립트 사용 (가장 쉬움) ⭐

```bash
# 1. 스크립트 실행 권한 부여
chmod +x setup_deepseek_mlx.sh

# 2. 실행 (모든 과정 자동화)
./setup_deepseek_mlx.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- 가상환경 생성
- 필요한 패키지 설치
- 모델 다운로드 및 변환
- 간단한 테스트 실행

### 방법 2: 수동 변환

```bash
# 1. mlx_lm 설치
pip install mlx-lm

# 2. MLX 형식으로 변환
python -m mlx_lm.convert \
    --hf-path "deepseek-ai/DeepSeek-MoE-16b-chat" \
    --mlx-path "./deepseek-16b-mlx-q4" \
    -q
```

### 방법 2: 검증된 모델 다운로드
- Hugging Face에서 통합된 MLX 모델 검색
- `mlx-community` 네임스페이스 확인

## 모델 파일 검증

변환 후 다음 스크립트로 검증:

```python
import mlx.core as mx
import json

# Config 확인
with open("config.json") as f:
    config = json.load(f)

expected_dim = config["hidden_size"]  # 2048

# Weight 확인
weights = mx.load("weights.safetensors")  # 또는 model.safetensors
q_proj = weights["model.layers.0.self_attn.q_proj.weight"]

print(f"Expected: ({expected_dim}, {expected_dim})")
print(f"Actual: {q_proj.shape}")

if q_proj.shape == (expected_dim, expected_dim):
    print("✅ 모델 파일 정상!")
else:
    print("❌ 모델 파일 문제 있음")
```

## 파일 크기 확인
- 16B 모델 4-bit 양자화: 약 8-10GB
- 현재 파일이 1-2GB라면 불완전한 파일

## 📚 추가 문서

- **완전한 설정 가이드**: `COMPLETE_SETUP_GUIDE.md` 참조
- **구현 요약**: `IMPLEMENTATION_SUMMARY.md` 참조
- **Python 채팅 테스트**: `chat_deepseek.py` 사용

