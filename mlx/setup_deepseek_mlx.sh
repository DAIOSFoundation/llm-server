#!/bin/bash

# --- 설정 변수 ---
MODEL_REPO="deepseek-ai/DeepSeek-MoE-16b-chat"
OUTPUT_DIR="./deepseek-16b-mlx-q4"

echo "=== DeepSeek-MoE-16b MLX 4-bit 변환 및 설정 시작 ==="

# 1. 파이썬 가상환경 생성 (권장)
if [ ! -d "venv" ]; then
    echo "[INFO] 가상환경(venv)을 생성합니다..."
    python3 -m venv venv
fi
source venv/bin/activate

# 2. 필수 패키지 설치
echo "[INFO] MLX 및 Hugging Face 패키지 설치 중..."
pip install -U pip
pip install -U mlx-lm huggingface_hub

# 3. 모델 다운로드 및 4-bit 양자화 변환
# -q: 4-bit 양자화 옵션
# --hf-path: 원본 Hugging Face 주소 (자동 다운로드 됨)
# --mlx-path: 변환된 모델이 저장될 로컬 경로
if [ -d "$OUTPUT_DIR" ]; then
    echo "[INFO] 이미 변환된 폴더가 존재합니다: $OUTPUT_DIR"
    read -p "다시 변환하시겠습니까? (y/n): " REBUILD
    if [ "$REBUILD" == "y" ]; then
        echo "[INFO] 기존 폴더 삭제 후 재변환..."
        rm -rf "$OUTPUT_DIR"
        python -m mlx_lm.convert --hf-path $MODEL_REPO --mlx-path $OUTPUT_DIR -q
    fi
else
    echo "[INFO] 변환(Quantization) 시작... (시간이 소요됩니다)"
    python -m mlx_lm.convert --hf-path $MODEL_REPO --mlx-path $OUTPUT_DIR -q
fi

# 4. 변환 결과 확인 (테스트)
echo "=== [TEST] 변환된 모델 테스트 ==="
python -m mlx_lm.generate --model $OUTPUT_DIR --prompt "Hello, introduce yourself." --max-tokens 50

echo "=== 완료되었습니다! ==="
echo "변환된 모델 위치: $OUTPUT_DIR"
echo "생성된 weights.safetensors 파일이 깨끗한 4bit 파일입니다."

