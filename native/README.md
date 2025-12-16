# Metal VRAM Monitor Native Module

macOS Metal API를 사용하여 VRAM 사용량을 모니터링하는 C++ native 모듈입니다.

## 빌드 방법

```bash
cd native
npm install
npm run build
```

또는 프로젝트 루트에서:

```bash
npm run build:native
```

## 사용 방법

```javascript
const { getVRAMInfo } = require('./native');

const info = getVRAMInfo();
console.log('VRAM Total:', info.total);
console.log('VRAM Used:', info.used);
```

## 요구사항

- Node.js 14+
- node-gyp
- macOS (Metal 지원)

## 구조

- `src/metal_vram.mm`: Objective-C++ 구현
- `index.js`: Node.js 래퍼
- `binding.gyp`: 빌드 설정

## 동작 방식

1. Metal API를 통해 시스템 기본 GPU 디바이스 가져오기
2. `recommendedMaxWorkingSetSize`로 VRAM 총량 확인
3. `currentAllocatedSize`로 현재 사용 중인 VRAM 확인
4. Node.js 객체로 반환
