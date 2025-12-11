// 공통 타입 정의

export interface ModelConfig {
  id: string;
  name: string;
  path: string;
  quantization: QuantizationType;
  quantizationLevel?: string;
  device: DeviceType;
  gpuBackend?: GPUBackend;
  inferenceConfig: InferenceConfig;
}

export type QuantizationType = 
  | 'Q4_0' | 'Q4_1' | 'Q5_0' | 'Q5_1' 
  | 'Q8_0' | 'F16' | 'F32';

export type DeviceType = 'CPU' | 'GPU';

export type GPUBackend = 'CUDA' | 'OpenCL';

export interface InferenceConfig {
  // Context 설정
  contextSize: number;        // n_ctx
  batchSize: number;          // n_batch
  
  // 생성 설정
  maxTokens: number;          // n_predict
  temperature: number;        // temperature
  topK: number;              // top_k
  topP: number;              // top_p
  repeatPenalty: number;     // repeat_penalty
  repeatLastN: number;       // repeat_last_n
  
  // 스레드 설정
  threads: number;            // n_threads
  threadsBatch?: number;     // n_threads_batch
  
  // 기타
  seed: number;              // seed (-1 for random)
  stopSequences: string[];   // stop sequences
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface ChatRequest {
  modelId: string;
  messages: ChatMessage[];
  stream?: boolean;
}

export interface ChatResponse {
  content: string;
  finishReason?: 'stop' | 'length' | 'error';
  tokensUsed?: number;
}

export type Language = 'ko' | 'en';

