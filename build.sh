#!/bin/bash

# llama.cpp 빌드를 위한 스크립트

set -e

echo "llama.cpp 빌드를 시작합니다..."

# 기존 디렉토리 삭제
if [ -d "llama.cpp" ]; then
    echo "기존 llama.cpp 디렉토리를 삭제합니다..."
    rm -rf llama.cpp
fi

# 1. llama.cpp 저장소 클론
echo "llama.cpp 저장소를 클론합니다..."
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. CMake를 사용하여 빌드 (Metal 지원 활성화)
echo "macOS용으로 빌드합니다 (CMake 사용)..."
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release

echo "빌드가 완료되었습니다."
echo "실행 파일은 ./build/bin/ 디렉토리에 있습니다."

cd ..

