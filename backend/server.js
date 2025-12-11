import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { ModelManager } from './model-manager.js';
import { InferenceEngine } from './inference-engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const CONFIG_PATH = join(__dirname, 'config.json');
const MODELS_PATH = join(__dirname, 'models.json');

// 설정 로드
let config = {};
if (existsSync(CONFIG_PATH)) {
  config = JSON.parse(readFileSync(CONFIG_PATH, 'utf-8'));
}

// 모델 관리자 초기화
const modelManager = new ModelManager(MODELS_PATH);
const inferenceEngine = new InferenceEngine();

// API 라우트

// 모델 목록 조회
app.get('/api/models', (req, res) => {
  try {
    const models = modelManager.getModels();
    res.json({ success: true, models });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 모델 추가
app.post('/api/models', (req, res) => {
  try {
    const { name, path, quantization, device, gpuBackend, inferenceConfig } = req.body;
    
    if (!name || !path) {
      return res.status(400).json({ 
        success: false, 
        error: '모델 이름과 경로가 필요합니다.' 
      });
    }

    if (!existsSync(path)) {
      return res.status(400).json({ 
        success: false, 
        error: '모델 파일을 찾을 수 없습니다.' 
      });
    }

    const model = modelManager.addModel({
      name,
      path,
      quantization: quantization || 'Q4_0',
      device: device || 'CPU',
      gpuBackend: gpuBackend || null,
      inferenceConfig: inferenceConfig || getDefaultInferenceConfig()
    });

    res.json({ success: true, model });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 모델 업데이트
app.put('/api/models/:id', (req, res) => {
  try {
    const { id } = req.params;
    const updates = req.body;
    
    const model = modelManager.updateModel(id, updates);
    if (!model) {
      return res.status(404).json({ success: false, error: '모델을 찾을 수 없습니다.' });
    }

    res.json({ success: true, model });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 모델 삭제
app.delete('/api/models/:id', (req, res) => {
  try {
    const { id } = req.params;
    const success = modelManager.deleteModel(id);
    
    if (!success) {
      return res.status(404).json({ success: false, error: '모델을 찾을 수 없습니다.' });
    }

    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 채팅 요청
app.post('/api/chat', async (req, res) => {
  try {
    const { modelId, messages, stream } = req.body;

    if (!modelId || !messages || !Array.isArray(messages)) {
      return res.status(400).json({ 
        success: false, 
        error: '모델 ID와 메시지가 필요합니다.' 
      });
    }

    const model = modelManager.getModel(modelId);
    if (!model) {
      return res.status(404).json({ 
        success: false, 
        error: '모델을 찾을 수 없습니다.' 
      });
    }

    if (stream) {
      // 스트리밍 응답
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const stream = await inferenceEngine.generateStream(model, messages);
      
      for await (const chunk of stream) {
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      // 일반 응답
      const response = await inferenceEngine.generate(model, messages);
      res.json({ success: true, ...response });
    }
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 시스템 정보 조회
app.get('/api/system', (req, res) => {
  try {
    const info = inferenceEngine.getSystemInfo();
    res.json({ success: true, ...info });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 기본 추론 설정
function getDefaultInferenceConfig() {
  return {
    contextSize: 2048,
    batchSize: 512,
    maxTokens: 512,
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repeatPenalty: 1.1,
    repeatLastN: 64,
    threads: 4,
    threadsBatch: 4,
    seed: -1,
    stopSequences: []
  };
}

const PORT = config.server?.port || 3001;
const HOST = config.server?.host || 'localhost';

app.listen(PORT, HOST, () => {
  console.log(`서버가 ${HOST}:${PORT}에서 실행 중입니다.`);
});

