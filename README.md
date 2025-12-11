# LLM Server - llama.cpp 기반 GGUF 모델 서빙 애플리케이션

llama.cpp를 사용하여 GGUF 모델을 서빙하는 애플리케이션입니다.

## 프로젝트 구조

```
llm-server/
├── frontend/          # React-Vite 프론트엔드
├── backend/           # llama.cpp 백엔드 서버
├── shared/            # 공통 타입 및 설정
└── README.md
```

## 기능

- CPU 및 GPU 하드웨어 가속 지원
- CUDA 및 OpenCL 설정 관리
- GGUF 모델 관리 (경로, 양자화 설정)
- 웹 기반 챗팅 인터페이스
- 모델별 추론 설정 관리

## 빌드 및 실행

### 백엔드 (macOS)

```bash
cd backend
./build.sh
./run.sh
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

