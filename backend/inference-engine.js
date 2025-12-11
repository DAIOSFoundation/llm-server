import { LlamaModel, LlamaContext, LlamaChatSession } from 'node-llama-cpp';
import { existsSync } from 'fs';

export class InferenceEngine {
  constructor() {
    this.loadedModels = new Map();
  }

  async loadModel(modelConfig) {
    const modelId = modelConfig.id;

    // 이미 로드된 모델이면 반환
    if (this.loadedModels.has(modelId)) {
      return this.loadedModels.get(modelId);
    }

    if (!existsSync(modelConfig.path)) {
      throw new Error(`모델 파일을 찾을 수 없습니다: ${modelConfig.path}`);
    }

    try {
      // GPU 백엔드 설정
      const gpuLayers = modelConfig.device === 'GPU' ? -1 : 0;
      
      // 모델 로드
      const model = new LlamaModel({
        modelPath: modelConfig.path,
        gpuLayers: gpuLayers,
        // CUDA/OpenCL 설정은 node-llama-cpp가 자동으로 처리
      });

      // 컨텍스트 생성
      const context = new LlamaContext({
        model: model,
        contextSize: modelConfig.inferenceConfig.contextSize || 2048,
        batchSize: modelConfig.inferenceConfig.batchSize || 512,
        threads: modelConfig.inferenceConfig.threads || 4,
        threadsBatch: modelConfig.inferenceConfig.threadsBatch || 4,
      });

      const session = new LlamaChatSession({
        context: context,
        systemPrompt: ''
      });

      const loadedModel = {
        model,
        context,
        session,
        config: modelConfig
      };

      this.loadedModels.set(modelId, loadedModel);
      return loadedModel;
    } catch (error) {
      throw new Error(`모델 로드 실패: ${error.message}`);
    }
  }

  async generate(modelConfig, messages) {
    const loaded = await this.loadModel(modelConfig);
    const { session, config } = loaded;

    // 메시지 포맷팅
    const formattedMessages = messages.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      text: msg.content
    }));

    // 추론 설정
    const inferenceConfig = config.inferenceConfig;
    
    let fullResponse = '';
    const response = await session.prompt(
      formattedMessages[formattedMessages.length - 1].text,
      {
        temperature: inferenceConfig.temperature,
        topK: inferenceConfig.topK,
        topP: inferenceConfig.topP,
        repeatPenalty: inferenceConfig.repeatPenalty,
        repeatLastN: inferenceConfig.repeatLastN,
        maxTokens: inferenceConfig.maxTokens,
        stopSequences: inferenceConfig.stopSequences
      }
    );

    return {
      content: response,
      finishReason: 'stop',
      tokensUsed: 0 // node-llama-cpp에서 토큰 수를 직접 제공하지 않을 수 있음
    };
  }

  async *generateStream(modelConfig, messages) {
    const loaded = await this.loadModel(modelConfig);
    const { session, config } = loaded;

    const formattedMessages = messages.map(msg => ({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      text: msg.content
    }));

    const inferenceConfig = config.inferenceConfig;
    
    // 스트리밍 생성 (node-llama-cpp의 스트리밍 API 사용)
    const lastMessage = formattedMessages[formattedMessages.length - 1];
    
    // 스트리밍은 node-llama-cpp의 API에 따라 구현 필요
    // 여기서는 간단한 예시로 구현
    const response = await session.prompt(
      lastMessage.text,
      {
        temperature: inferenceConfig.temperature,
        topK: inferenceConfig.topK,
        topP: inferenceConfig.topP,
        repeatPenalty: inferenceConfig.repeatPenalty,
        repeatLastN: inferenceConfig.repeatLastN,
        maxTokens: inferenceConfig.maxTokens,
        stopSequences: inferenceConfig.stopSequences
      }
    );

    // 응답을 청크로 분할하여 스트리밍
    const chunkSize = 10;
    for (let i = 0; i < response.length; i += chunkSize) {
      yield {
        content: response.slice(i, i + chunkSize),
        finishReason: i + chunkSize >= response.length ? 'stop' : null
      };
    }
  }

  unloadModel(modelId) {
    if (this.loadedModels.has(modelId)) {
      const loaded = this.loadedModels.get(modelId);
      // 리소스 정리
      if (loaded.context) {
        // node-llama-cpp의 정리 메서드 호출
      }
      this.loadedModels.delete(modelId);
    }
  }

  getSystemInfo() {
    // 시스템 정보 조회 (CPU 코어 수, GPU 정보 등)
    const os = require('os');
    
    return {
      cpuCores: os.cpus().length,
      platform: process.platform,
      arch: process.arch,
      // GPU 정보는 별도 라이브러리 필요
    };
  }
}

