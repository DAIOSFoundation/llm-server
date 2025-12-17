# 모델 설정 빠른 시작

## 🚀 가장 빠른 방법

```bash
# 1. 자동화 스크립트 실행
chmod +x setup_deepseek_mlx.sh
./setup_deepseek_mlx.sh

# 2. Python으로 테스트
source venv/bin/activate
python chat_deepseek.py

# 3. C++ 프로젝트에 모델 복사
cp -r ./deepseek-16b-mlx-q4/* ./models/deepseek-moe-16b-chat-mlx-q4_0/

# 4. C++ 서버 실행
npm run build
node test-server-temp.js
```

## 📖 상세 가이드

- **완전한 설정 가이드**: `COMPLETE_SETUP_GUIDE.md`
- **모델 설정 요약**: `MODEL_SETUP_GUIDE.md`
- **구현 완료 요약**: `IMPLEMENTATION_SUMMARY.md`

## ✅ 체크리스트

- [ ] 모델 변환 완료 (`setup_deepseek_mlx.sh`)
- [ ] Python 테스트 성공 (`chat_deepseek.py`)
- [ ] 가중치 검증 통과 (q_proj: 2048x2048)
- [ ] C++ 서버 정상 작동

## ⚠️ 중요

현재 모델 파일이 불완전합니다. 위 스크립트로 정상적인 모델을 생성하세요.
