#!/bin/bash

# llama.cpp 빌드 스크립트 (macOS)

set -e

echo "llama.cpp 빌드를 시작합니다..."

# 디렉토리 확인
if [ ! -d "llama.cpp" ]; then
    echo "llama.cpp 저장소를 클론합니다..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# macOS 빌드
echo "macOS용으로 빌드합니다..."

# Metal (GPU) 지원 빌드
make clean
make -j$(sysctl -n hw.ncpu) \
    LLAMA_METAL=1 \
    LLAMA_OPENBLAS=0

echo "빌드 완료!"
echo "실행 파일: ./main"

