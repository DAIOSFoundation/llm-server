const http = require('http');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { URL } = require('url');

// C++ native 모듈 사용 시도
let MlxServerNative = null;
try {
  MlxServerNative = require('./native');
  console.log('[MLX Server] C++ native module loaded successfully');
} catch (error) {
  console.error('[MLX Server] ❌ C++ native module not available:', error.message);
  throw new Error('MLX C++ native module is required. Please build it with: cd mlx && npm run build');
}

class MlxServer {
  constructor(modelConfig) {
    this.modelConfig = modelConfig;
    this.modelPath = modelConfig.modelPath;
    this.modelDir = path.join(__dirname, 'models', this.modelPath);
    this.server = null;
    this.port = 8081; // MLX 서버는 8081 포트 사용
    this.isModelLoaded = false;
    this.modelMeta = null;
    this.loadStartTime = null;
    this.nativeServer = null; // C++ native 모듈 인스턴스
  }

  async loadModel() {
    console.log(`[MLX] Loading model from: ${this.modelDir}`);
    
    if (!fs.existsSync(this.modelDir)) {
      const error = `Model directory not found: ${this.modelDir}`;
      console.error(`[MLX] ${error}`);
      throw new Error(error);
    }

    const configPath = path.join(this.modelDir, 'config.json');
    if (!fs.existsSync(configPath)) {
      const error = `Model config.json not found: ${configPath}`;
      console.error(`[MLX] ${error}`);
      throw new Error(error);
    }

    this.loadStartTime = Date.now();
    try {
      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      this.modelMeta = {
        id: this.modelPath,
        path: this.modelDir,
        format: 'mlx',
        context_length: config.max_position_embeddings || this.modelConfig.contextSize || 2048,
        architecture: config.architectures?.[0] || 'unknown',
        model_type: config.model_type || 'unknown',
      };

      // C++ native 모듈 인스턴스 생성 및 모델 로드
      if (MlxServerNative) {
        try {
          console.log(`[MLX] [LOG] Step 1: Creating C++ native module instance for model: ${this.modelPath}`);
          console.log(`[MLX] [LOG] Step 1.1: Model directory: ${this.modelDir}`);
          console.log(`[MLX] [LOG] Step 1.2: Calling MlxServerNative constructor...`);
          const startTime = Date.now();
          this.nativeServer = new MlxServerNative(this.modelDir);
          const loadTime = Date.now() - startTime;
          console.log(`[MLX] [LOG] Step 1.3: Constructor completed in ${loadTime}ms`);
          console.log(`[MLX] ✅ C++ native module instance created successfully`);
        } catch (error) {
          console.error(`[MLX] ❌ Failed to create native server instance:`, error);
          console.error(`[MLX]    Error message:`, error.message);
          console.error(`[MLX]    Error stack:`, error.stack);
          // 모델 메타데이터는 로드했으므로 계속 진행하지만, nativeServer는 null로 남음
          this.nativeServer = null;
        }
      } else {
        console.error(`[MLX] ❌ MlxServerNative class is not available`);
        this.nativeServer = null;
      }

      console.log(`[MLX] [LOG] Step 2: Setting isModelLoaded flag`);
      this.isModelLoaded = true;
      console.log(`[MLX] [LOG] Step 3: Model metadata loaded: ${this.modelPath}`);
      console.log(`[MLX] Model metadata loaded: ${this.modelPath}`);
      console.log(`[MLX] Model context length: ${this.modelMeta.context_length}`);
    } catch (error) {
      console.error(`[MLX] Failed to load model config:`, error);
      throw error;
    }
  }

  async start() {
    try {
      console.log(`[MLX] [LOG] start() called - isModelLoaded: ${this.isModelLoaded}`);
      if (!this.isModelLoaded) {
        console.log(`[MLX] [LOG] Model not loaded, calling loadModel()...`);
        await this.loadModel();
        console.log(`[MLX] [LOG] loadModel() completed`);
      } else {
        console.log(`[MLX] [LOG] Model already loaded, skipping loadModel()`);
      }

      this.server = http.createServer((req, res) => {
        this.handleRequest(req, res);
      });

      return new Promise((resolve, reject) => {
        this.server.on('error', (err) => {
          if (err.code === 'EADDRINUSE') {
            console.error(`[MLX] Port ${this.port} is already in use. Please stop the existing server.`);
            reject(new Error(`Port ${this.port} is already in use`));
          } else {
            console.error(`[MLX] Server error:`, err);
            reject(err);
          }
        });

        console.log(`[MLX] [LOG] Step 5: Starting HTTP server on port ${this.port}...`);
        this.server.listen(this.port, () => {
          console.log(`[MLX] [LOG] Step 5.1: HTTP server listen() callback called`);
          console.log(`[MLX] Server started on port ${this.port}`);
          console.log(`[MLX] Model loaded: ${this.modelPath}`);
          console.log(`[MLX] [LOG] Step 5.2: Server fully started and ready`);
          resolve(this.server);
        });
        console.log(`[MLX] [LOG] Step 5.3: listen() called, waiting for callback...`);
      });
    } catch (error) {
      console.error(`[MLX] Failed to start server:`, error);
      throw error;
    }
  }

  async handleRequest(req, res) {
    // CORS 헤더 설정
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-LLM-UI-Auth');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathname = url.pathname;

    try {
      // Health check
      if ((pathname === '/health' || pathname === '/v1/health') && req.method === 'GET') {
        await this.handleHealth(req, res);
      }
      // Metrics
      else if (pathname === '/metrics' && req.method === 'GET') {
        await this.handleMetrics(req, res);
      }
      // Metrics Stream (SSE)
      else if (pathname === '/metrics/stream' && req.method === 'GET') {
        await this.handleMetricsStream(req, res);
      }
      // Props
      else if (pathname === '/props' && req.method === 'GET') {
        await this.handleProps(req, res);
      }
      else if (pathname === '/props' && req.method === 'POST') {
        await this.handlePostProps(req, res);
      }
      // Models
      else if ((pathname === '/models' || pathname === '/v1/models') && req.method === 'GET') {
        await this.handleModels(req, res);
      }
      // Model management
      else if (pathname === '/models/load' && req.method === 'POST') {
        await this.handleModelLoad(req, res);
      }
      else if (pathname === '/models/unload' && req.method === 'POST') {
        await this.handleModelUnload(req, res);
      }
      else if (pathname === '/models/status' && req.method === 'POST') {
        await this.handleModelStatus(req, res);
      }
      else if (pathname === '/models/config' && req.method === 'GET') {
        await this.handleModelConfigGet(req, res);
      }
      else if (pathname === '/models/config' && req.method === 'POST') {
        await this.handleModelConfigPost(req, res);
      }
      // Completion
      else if (pathname === '/completion' && req.method === 'POST') {
        await this.handleCompletion(req, res);
      }
      else if (pathname === '/completions' && req.method === 'POST') {
        await this.handleCompletion(req, res);
      }
      else if (pathname === '/v1/completions' && req.method === 'POST') {
        await this.handleCompletion(req, res);
      }
      // Chat completions
      else if (pathname === '/chat/completions' && req.method === 'POST') {
        await this.handleChatCompletions(req, res);
      }
      else if (pathname === '/v1/chat/completions' && req.method === 'POST') {
        await this.handleChatCompletions(req, res);
      }
      // Tokenize
      else if (pathname === '/tokenize' && req.method === 'POST') {
        await this.handleTokenize(req, res);
      }
      // Detokenize
      else if (pathname === '/detokenize' && req.method === 'POST') {
        await this.handleDetokenize(req, res);
      }
      // Logs Stream (SSE)
      else if (pathname === '/logs/stream' && req.method === 'GET') {
        await this.handleLogsStream(req, res);
      }
      // Apply template
      else if (pathname === '/apply-template' && req.method === 'POST') {
        await this.handleApplyTemplate(req, res);
      }
      // MLX model verification
      else if (pathname === '/mlx-verify' && req.method === 'POST') {
        await this.handleMlxVerify(req, res);
      }
      else {
        console.log(`[MLX] 404: ${req.method} ${pathname}`);
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found' }));
      }
    } catch (error) {
      console.error(`[MLX] Error handling ${req.method} ${pathname}:`, error);
      const statusCode = error.statusCode || 500;
      res.writeHead(statusCode, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: error.message || String(error) }));
    }
  }

  async handleHealth(req, res) {
    if (!this.isModelLoaded) {
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: {
          code: 503,
          message: 'Loading model',
          type: 'unavailable_error'
        }
      }));
    } else {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok' }));
    }
  }

  async handleMetrics(req, res) {
    // Metal API를 통해 VRAM 정보 가져오기 시도
    let vramTotal = 0;
    let vramUsed = 0;
    let vramFree = 0;

    try {
      const metalVram = require('../native');
      const vramInfo = metalVram.getVRAMInfo();
      if (!vramInfo.error) {
        vramTotal = vramInfo.total || 0;
        vramUsed = vramInfo.used || 0;
        vramFree = Math.max(0, vramTotal - vramUsed);
      }
    } catch (e) {
      // Metal VRAM 정보를 가져올 수 없으면 기본값 사용
    }

    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end(`llamacpp:vram_total_bytes ${vramTotal}
llamacpp:vram_used_bytes ${vramUsed}
llamacpp:vram_free_bytes ${vramFree}
llamacpp:predicted_tokens_seconds 0
`);
  }

  async handleMetricsStream(req, res) {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const modelParam = url.searchParams.get('model');
      const intervalMs = Math.max(250, Math.min(10000, parseInt(url.searchParams.get('interval_ms') || '1000', 10)));
      
      // 모델 파라미터 확인 (선택사항, 현재 로드된 모델과 일치하는지 확인)
      if (modelParam && modelParam !== this.modelPath) {
        console.log(`[MLX] Metrics stream requested for model: ${modelParam}, current: ${this.modelPath}`);
        // 모델이 다르더라도 계속 진행 (MLX 서버는 단일 모델만 지원)
      }
      
      console.log(`[MLX] Metrics stream started, interval: ${intervalMs}ms`);

      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      let intervalId = null;
      let isClosed = false;

      const sendMetrics = async () => {
        if (isClosed || res.writableEnded) return;

        try {
          // Metal API를 통해 VRAM 정보 가져오기 시도
          let vramTotal = 0;
          let vramUsed = 0;
          let vramFree = 0;

          try {
            const metalVram = require('../native');
            const vramInfo = metalVram.getVRAMInfo();
            if (!vramInfo.error) {
              vramTotal = vramInfo.total || 0;
              vramUsed = vramInfo.used || 0;
              vramFree = Math.max(0, vramTotal - vramUsed);
            }
          } catch (e) {
            // Metal VRAM 정보를 가져올 수 없으면 기본값 사용
            console.warn('[MLX] Could not get VRAM info:', e.message);
          }

          const metrics = {
            vram_total_bytes: vramTotal,
            vram_used_bytes: vramUsed,
            vram_free_bytes: vramFree,
            predicted_tokens_seconds: 0,
          };

          if (!isClosed && !res.writableEnded) {
            res.write(`event: metrics\n`);
            res.write(`data: ${JSON.stringify(metrics)}\n\n`);
          }
        } catch (error) {
          console.error('[MLX] Error sending metrics:', error);
          if (!isClosed && !res.writableEnded) {
            try {
              res.write(`event: error\n`);
              res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
            } catch (e) {
              // 연결이 이미 종료된 경우 무시
            }
          }
        }
      };

      // 즉시 첫 메트릭 전송
      await sendMetrics();

      // 주기적으로 메트릭 전송
      intervalId = setInterval(() => {
        if (!isClosed && !res.writableEnded) {
          sendMetrics();
        } else {
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
        }
      }, intervalMs);

      // 클라이언트 연결 종료 감지
      req.on('close', () => {
        isClosed = true;
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
        console.log('[MLX] Metrics stream closed');
      });

      req.on('error', (error) => {
        console.error('[MLX] Metrics stream request error:', error);
        isClosed = true;
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      });
    } catch (error) {
      console.error('[MLX] Error in handleMetricsStream:', error);
      if (!res.writableEnded) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message || String(error) }));
      }
    }
  }

  async handleProps(req, res) {
    const defaultParams = {
      n_predict: this.modelConfig.maxTokens || -1,
      seed: -1,
      temperature: this.modelConfig.temperature || 0.7,
      dynatemp_range: 0.0,
      dynatemp_exponent: 1.0,
      top_k: this.modelConfig.topK || 40,
      top_p: this.modelConfig.topP || 0.95,
      min_p: this.modelConfig.minP || 0.05,
      xtc_probability: 0.0,
      xtc_threshold: 0.1,
      typical_p: this.modelConfig.typicalP || 1.0,
      repeat_last_n: this.modelConfig.repeatLastN || 128,
      repeat_penalty: this.modelConfig.repeatPenalty || 1.2,
      presence_penalty: this.modelConfig.presencePenalty || 0.0,
      frequency_penalty: this.modelConfig.frequencyPenalty || 0.0,
      dry_multiplier: this.modelConfig.dryMultiplier || 0.0,
      dry_base: this.modelConfig.dryBase || 1.75,
      dry_allowed_length: this.modelConfig.dryAllowedLength || 2,
      dry_penalty_last_n: this.modelConfig.dryPenaltyLastN || -1,
      mirostat: this.modelConfig.mirostatMode || 0,
      mirostat_tau: this.modelConfig.mirostatTau || 5.0,
      mirostat_eta: this.modelConfig.mirostatEta || 0.1,
      stop: [],
      max_tokens: this.modelConfig.maxTokens || -1,
      n_keep: 0,
      n_discard: 0,
      ignore_eos: false,
      stream: true,
      n_probs: 0,
      min_keep: 0,
      grammar: '',
      samplers: ['dry', 'top_k', 'typ_p', 'top_p', 'min_p', 'xtc', 'temperature'],
      'speculative.n_max': 16,
      'speculative.n_min': 5,
      'speculative.p_min': 0.9,
      timings_per_token: false,
    };

    const data = {
      default_generation_settings: {
        id: 0,
        id_task: -1,
        n_ctx: this.modelConfig.contextSize || 2048,
        speculative: false,
        is_processing: false,
        params: defaultParams,
        prompt: '',
        next_token: {
          has_next_token: true,
          has_new_line: false,
          n_remain: -1,
          n_decoded: 0,
          stopping_word: '',
        },
      },
      total_slots: 1,
      model_alias: this.modelPath,
      model_path: this.modelDir,
      modalities: {
        vision: false,
        audio: false,
      },
      endpoint_slots: false,
      endpoint_props: true,
      endpoint_metrics: true,
      webui: false,
      chat_template: this.getChatTemplate(),
      bos_token: '<|begin_of_text|>',
      eos_token: '<|end_of_text|>',
      build_info: 'mlx-server-1.0.0',
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  }

  async handlePostProps(req, res) {
    // Props 업데이트는 현재 지원하지 않음
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true }));
  }

  async handleModels(req, res) {
    const data = {
      data: [{
        id: this.modelPath,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'mlx',
        meta: this.modelMeta,
        status: {
          value: this.isModelLoaded ? 'loaded' : 'unloaded',
        },
      }],
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  }

  async handleModelLoad(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        const modelId = payload.model || this.modelPath;
        
        if (modelId === this.modelPath) {
          if (!this.isModelLoaded) {
            this.loadModel();
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true }));
        } else {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Model not found' }));
        }
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleModelUnload(req, res) {
    this.isModelLoaded = false;
    this.modelMeta = null;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true }));
  }

  async handleModelStatus(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        const modelId = payload.model || this.modelPath;
        
        const status = {
          value: this.isModelLoaded ? 'loaded' : 'unloaded',
        };

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status }));
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleModelConfigGet(req, res) {
    const config = {
      [this.modelPath]: {
        contextSize: this.modelConfig.contextSize || 2048,
        gpuLayers: 0, // MLX는 항상 GPU 사용
      },
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(config));
  }

  async handleModelConfigPost(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        if (payload.model === this.modelPath && payload.config) {
          // 설정 업데이트
          Object.assign(this.modelConfig, payload.config);
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleCompletion(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        const { prompt, stream = true, temperature, top_k, top_p, min_p, 
                typical_p, tfs_z, repeat_penalty, repeat_last_n, 
                presence_penalty, frequency_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
                mirostat, mirostat_tau, mirostat_eta,
                n_predict, stop, seed } = payload;

        const options = {
          temperature: temperature ?? this.modelConfig.temperature ?? 0.7,
          top_k: top_k ?? this.modelConfig.topK ?? 40,
          top_p: top_p ?? this.modelConfig.topP ?? 0.95,
          min_p: min_p ?? this.modelConfig.minP ?? 0.05,
          typical_p: typical_p ?? this.modelConfig.typicalP ?? 1.0,
          tfs_z: tfs_z ?? this.modelConfig.tfsZ ?? 1.0,
          repeat_penalty: repeat_penalty ?? this.modelConfig.repeatPenalty ?? 1.2,
          repeat_last_n: repeat_last_n ?? this.modelConfig.repeatLastN ?? 128,
          presence_penalty: presence_penalty ?? this.modelConfig.presencePenalty ?? 0.0,
          frequency_penalty: frequency_penalty ?? this.modelConfig.frequencyPenalty ?? 0.0,
          dry_multiplier: dry_multiplier ?? this.modelConfig.dryMultiplier ?? 0.0,
          dry_base: dry_base ?? this.modelConfig.dryBase ?? 1.75,
          dry_allowed_length: dry_allowed_length ?? this.modelConfig.dryAllowedLength ?? 2,
          dry_penalty_last_n: dry_penalty_last_n ?? this.modelConfig.dryPenaltyLastN ?? -1,
          mirostat: mirostat ?? this.modelConfig.mirostatMode ?? 0,
          mirostat_tau: mirostat_tau ?? this.modelConfig.mirostatTau ?? 5.0,
          mirostat_eta: mirostat_eta ?? this.modelConfig.mirostatEta ?? 0.1,
          max_tokens: n_predict ?? this.modelConfig.maxTokens ?? 600,
          stop: stop || [],
          seed: seed ?? -1,
        };

        if (stream) {
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });
          await this.streamCompletion(prompt, options, res);
        } else {
          await this.nonStreamCompletion(prompt, options, res);
        }
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleChatCompletions(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        const { messages, stream = true, model, temperature, top_p, top_k, min_p,
                typical_p, tfs_z, repeat_penalty, repeat_last_n,
                presence_penalty, frequency_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
                mirostat, mirostat_tau, mirostat_eta,
                max_tokens, stop, seed } = payload;

        // 메시지를 프롬프트로 변환 (비동기)
        const prompt = await this.messagesToPrompt(messages);

        const options = {
          temperature: temperature ?? this.modelConfig.temperature ?? 0.7,
          top_k: top_k ?? this.modelConfig.topK ?? 40,
          top_p: top_p ?? this.modelConfig.topP ?? 0.95,
          min_p: min_p ?? this.modelConfig.minP ?? 0.05,
          typical_p: typical_p ?? this.modelConfig.typicalP ?? 1.0,
          tfs_z: tfs_z ?? this.modelConfig.tfsZ ?? 1.0,
          repeat_penalty: repeat_penalty ?? this.modelConfig.repeatPenalty ?? 1.2,
          repeat_last_n: repeat_last_n ?? this.modelConfig.repeatLastN ?? 128,
          presence_penalty: presence_penalty ?? this.modelConfig.presencePenalty ?? 0.0,
          frequency_penalty: frequency_penalty ?? this.modelConfig.frequencyPenalty ?? 0.0,
          dry_multiplier: dry_multiplier ?? this.modelConfig.dryMultiplier ?? 0.0,
          dry_base: dry_base ?? this.modelConfig.dryBase ?? 1.75,
          dry_allowed_length: dry_allowed_length ?? this.modelConfig.dryAllowedLength ?? 2,
          dry_penalty_last_n: dry_penalty_last_n ?? this.modelConfig.dryPenaltyLastN ?? -1,
          mirostat: mirostat ?? this.modelConfig.mirostatMode ?? 0,
          mirostat_tau: mirostat_tau ?? this.modelConfig.mirostatTau ?? 5.0,
          mirostat_eta: mirostat_eta ?? this.modelConfig.mirostatEta ?? 0.1,
          max_tokens: max_tokens ?? this.modelConfig.maxTokens ?? 600,
          stop: stop || [],
          seed: seed ?? -1,
        };

        if (stream) {
          res.writeHead(200, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });
          await this.streamCompletion(prompt, options, res);
        } else {
          await this.nonStreamCompletion(prompt, options, res);
        }
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleTokenize(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        // llama.cpp 형식: { model, content, add_special, parse_special }
        // MLX는 model 파라미터 무시 (현재 로드된 모델 사용)
        const content = payload.content || payload.text || '';
        
        if (!content) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Content is required' }));
          return;
        }
        
        console.log(`[MLX] Tokenizing content, length: ${content.length}`);
        
        // C++ native 모듈만 사용 (Python fallback 제거)
        if (!this.nativeServer) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'MLX C++ native module instance is not available. Model may not be loaded.' }));
          return;
        }

        try {
          const tokens = this.nativeServer.tokenize(content);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ tokens: Array.isArray(tokens) ? tokens : Array.from(tokens) }));
        } catch (error) {
          console.error('[MLX] Native tokenize error:', error);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Tokenization failed: ${error.message}` }));
        }
      } catch (error) {
        console.error('[MLX] Tokenize error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleDetokenize(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        const { tokens } = payload;
        
        if (!Array.isArray(tokens)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Tokens must be an array' }));
          return;
        }
        
        // C++ native 모듈만 사용 (Python fallback 제거)
        if (!this.nativeServer) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'MLX C++ native module instance is not available. Model may not be loaded.' }));
          return;
        }

        try {
          const content = this.nativeServer.detokenize(tokens);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ content }));
        } catch (error) {
          console.error('[MLX] Native detokenize error:', error);
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Detokenization failed: ${error.message}` }));
        }
      } catch (error) {
        console.error('[MLX] Detokenize error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleApplyTemplate(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        const { messages, template } = payload;
        
        // 채팅 템플릿 적용
        const prompt = this.applyChatTemplate(messages, template);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ prompt }));
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: error.message }));
      }
    });
  }

  async handleMlxVerify(req, res) {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        const modelId = payload.model;
        
        if (!modelId || typeof modelId !== 'string') {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ exists: false, error: 'Invalid modelId' }));
          return;
        }
        
        // 여러 경로 시도
        const possiblePaths = [
          path.join(__dirname, 'models', modelId),
          path.resolve(__dirname, '..', 'mlx', 'models', modelId),
          path.resolve(process.cwd(), 'mlx', 'models', modelId),
        ];
        
        let foundPath = null;
        for (const modelDir of possiblePaths) {
          console.log(`[MLX Verify] Checking path: ${modelDir}`);
          if (fs.existsSync(modelDir)) {
            const stats = fs.statSync(modelDir);
            if (stats.isDirectory()) {
              const configPath = path.join(modelDir, 'config.json');
              if (fs.existsSync(configPath)) {
                foundPath = modelDir;
                console.log(`[MLX Verify] Found model at: ${foundPath}`);
                break;
              }
            }
          }
        }
        
        const exists = foundPath !== null;
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ exists, path: foundPath }));
      } catch (error) {
        console.error('[MLX Verify] Error:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ exists: false, error: error.message }));
      }
    });
  }

  async messagesToPrompt(messages) {
    // 채팅 템플릿 파일이 있으면 사용, 없으면 기본 형식 사용
    const templatePath = path.join(this.modelDir, 'chat_template.jinja');
    if (fs.existsSync(templatePath)) {
      return await this.applyChatTemplateWithJinja(messages, templatePath);
    }
    
    // 기본 템플릿 적용
    return this.messagesToPromptFallback(messages);
  }

  async applyChatTemplateWithJinja(messages, templatePath) {
    // Python을 통해 Jinja 템플릿 적용
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, 'apply_template.py');
      this.createApplyTemplateScript(scriptPath);
      
      const pythonProcess = spawn('python3', [
        scriptPath,
        '--template', templatePath,
        '--messages', JSON.stringify(messages),
      ]);
      
      let output = '';
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output);
            resolve(result.prompt || '');
          } catch (e) {
            // 파싱 실패 시 기본 형식 사용
            resolve(this.messagesToPromptFallback(messages));
          }
        } else {
          // 실패 시 기본 형식 사용
          resolve(this.messagesToPromptFallback(messages));
        }
      });
    });
  }

  messagesToPromptFallback(messages) {
    return messages.map(msg => {
      const role = msg.role === 'user' ? 'User' : 'Assistant';
      return `${role}: ${msg.content}`;
    }).join('\n\n') + '\n\nAssistant:';
  }

  applyChatTemplate(messages, template) {
    // 템플릿이 제공되면 사용, 없으면 모델의 템플릿 사용
    if (template) {
      // 제공된 템플릿 사용 (간단한 구현)
      return this.messagesToPromptFallback(messages);
    }
    return this.messagesToPrompt(messages);
  }

  getChatTemplate() {
    // 모델의 채팅 템플릿 반환
    const templatePath = path.join(this.modelDir, 'chat_template.jinja');
    if (fs.existsSync(templatePath)) {
      return fs.readFileSync(templatePath, 'utf-8');
    }
    // 기본 채팅 템플릿 반환
    return '{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}Assistant:';
  }

  async tokenize(content) {
    // Python 스크립트를 통해 토큰화
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, 'mlx_tokenize_script.py');
      this.createTokenizeScript(scriptPath);
      
      const pythonProcess = spawn('python3', [scriptPath, '--model', this.modelDir, '--text', content]);
      let output = '';
      let errorOutput = '';
      
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output.trim());
            if (result.error) {
              reject(new Error(result.error));
            } else {
              resolve(result.tokens || []);
            }
          } catch (e) {
            console.error(`[MLX] Failed to parse tokenize output:`, output);
            console.error(`[MLX] Error output:`, errorOutput);
            reject(new Error(`Failed to parse tokenize output: ${e.message}`));
          }
        } else {
          console.error(`[MLX] Tokenize script failed with code ${code}`);
          console.error(`[MLX] Error output:`, errorOutput);
          reject(new Error(`Tokenize failed with code ${code}: ${errorOutput}`));
        }
      });
      
      pythonProcess.on('error', (error) => {
        console.error(`[MLX] Failed to spawn tokenize process:`, error);
        reject(new Error(`Failed to spawn tokenize process: ${error.message}`));
      });
    });
  }

  async detokenize(tokens) {
    // Python 스크립트를 통해 디토큰화
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(__dirname, 'detokenize.py');
      this.createDetokenizeScript(scriptPath);
      
      const pythonProcess = spawn('python3', [
        scriptPath, 
        '--model', this.modelDir, 
        '--tokens', JSON.stringify(tokens)
      ]);
      let output = '';
      
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output);
            resolve(result.content || '');
          } catch (e) {
            reject(new Error('Failed to parse detokenize output'));
          }
        } else {
          reject(new Error(`Detokenize failed with code ${code}`));
        }
      });
    });
  }

  async nonStreamCompletion(prompt, options, res) {
    // 비스트리밍 응답
    let fullContent = '';
    let stopped = false;
    
    await new Promise((resolve, reject) => {
      const mockRes = {
        write: (data) => {
          try {
            const parsed = JSON.parse(data.replace('data: ', ''));
            if (parsed.content) {
              fullContent += parsed.content;
            }
            if (parsed.stop) {
              stopped = true;
              resolve();
            }
          } catch (e) {
            // 무시
          }
        },
        writableEnded: false,
        end: () => {
          if (!stopped) {
            stopped = true;
          }
          resolve();
        },
      };

      this.streamCompletion(prompt, options, mockRes).then(() => {
        if (!stopped) resolve();
      }).catch(reject);
    });

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      content: fullContent,
      stop: true,
      generation_settings: {
        n_ctx: this.modelConfig.contextSize || 2048,
        n_keep: 0,
        n_discard: 0,
        n_predict: options.max_tokens || -1,
        temperature: options.temperature,
        top_k: options.top_k,
        top_p: options.top_p,
        min_p: options.min_p,
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n,
      },
      prompt: prompt,
      tokens_predicted: 0,
      tokens_evaluated: 0,
      truncated: false,
    }));
  }

  async streamCompletion(prompt, options, res) {
    // C++ native 모듈만 사용 (Python fallback 제거)
    if (!this.nativeServer) {
      const errorMsg = 'MLX C++ native module instance is not available. Model may not be loaded.';
      console.error(`[MLX] ${errorMsg}`);
      if (!res.writableEnded) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: errorMsg }));
      }
      return;
    }

    try {
      await this.nativeServer.generateStream(
        prompt,
        {
          temperature: options.temperature || 0.7,
          topK: options.top_k || 40,
          topP: options.top_p || 0.95,
          minP: options.min_p || 0.05,
          typicalP: options.typical_p || 1.0,
          tfsZ: options.tfs_z || 1.0,
          repeatPenalty: options.repeat_penalty || 1.2,
          repeatLastN: options.repeat_last_n || 128,
          presencePenalty: options.presence_penalty || 0.0,
          frequencyPenalty: options.frequency_penalty || 0.0,
          dryMultiplier: options.dry_multiplier || 0.0,
          dryBase: options.dry_base || 1.75,
          dryAllowedLength: options.dry_allowed_length || 2,
          dryPenaltyLastN: options.dry_penalty_last_n || -1,
          mirostat: options.mirostat || 0,
          mirostatTau: options.mirostat_tau || 5.0,
          mirostatEta: options.mirostat_eta || 0.1,
          maxTokens: options.max_tokens || 600,
          stop: options.stop || [],
          seed: options.seed || -1,
        },
        (data) => {
          if (!res.writableEnded) {
            if (data.token) {
              res.write(`data: ${JSON.stringify({ content: data.token })}\n\n`);
            }
            if (data.stop) {
              res.write(`data: ${JSON.stringify({ stop: true })}\n\n`);
              res.end();
            }
            if (data.error) {
              res.write(`data: ${JSON.stringify({ error: data.error })}\n\n`);
              res.end();
            }
          }
        },
        (error) => {
          console.error('[MLX] Native module error:', error);
          if (!res.writableEnded) {
            res.write(`data: ${JSON.stringify({ error: String(error) })}\n\n`);
            res.write(`data: ${JSON.stringify({ stop: true })}\n\n`);
            res.end();
          }
        },
        () => {
          if (!res.writableEnded) {
            res.write(`data: ${JSON.stringify({ stop: true })}\n\n`);
            res.end();
          }
        }
      );
    } catch (error) {
      console.error('[MLX] Native module initialization error:', error);
      if (!res.writableEnded) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: `Failed to initialize MLX C++ native module: ${error.message}` }));
      }
    }
  }

  createInferenceScript(scriptPath) {
    const scriptContent = `#!/usr/bin/env python3
import json
import sys
import argparse
import mlx.core as mx
from mlx_lm import load, stream_generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--options', required=True)
    
    args = parser.parse_args()
    
    try:
        options = json.loads(args.options)
        
        # 모델 로드
        print(json.dumps({'status': 'loading_model'}), flush=True, file=sys.stderr)
        model, tokenizer = load(args.model)
        print(json.dumps({'status': 'model_loaded'}), flush=True, file=sys.stderr)
        
        # 생성 파라미터 설정 (MLX가 지원하는 파라미터만 사용)
        # MLX_LM은 일부 고급 샘플링 옵션을 지원하지 않을 수 있음
        generate_kwargs = {
            'temp': options.get('temperature', 0.7),
            'top_k': options.get('top_k', 40),
            'top_p': options.get('top_p', 0.95),
            'min_p': options.get('min_p', 0.05),
            'repetition_penalty': options.get('repeat_penalty', 1.2),
            'repetition_context_size': options.get('repeat_last_n', 128),
            'max_tokens': options.get('max_tokens', 600),
        }
        
        # 지원되지 않는 옵션에 대한 경고 (로그만 출력)
        unsupported = []
        if options.get('typical_p', 1.0) != 1.0:
            unsupported.append('typical_p')
        if options.get('tfs_z', 1.0) != 1.0:
            unsupported.append('tfs_z')
        if options.get('presence_penalty', 0.0) != 0.0:
            unsupported.append('presence_penalty')
        if options.get('frequency_penalty', 0.0) != 0.0:
            unsupported.append('frequency_penalty')
        if options.get('dry_multiplier', 0.0) != 0.0:
            unsupported.append('dry_multiplier')
        if options.get('mirostat', 0) != 0:
            unsupported.append('mirostat')
        
        if unsupported:
            print(json.dumps({'warning': f'Unsupported options (using defaults): {", ".join(unsupported)}'}), flush=True, file=sys.stderr)
        
        # Stop 토큰 설정
        stop_tokens = options.get('stop', [])
        
        # 스트리밍 생성
        print(json.dumps({'status': 'generating'}), flush=True, file=sys.stderr)
        generated_text = ''
        for token_str in stream_generate(
            model, tokenizer, 
            prompt=args.prompt,
            **generate_kwargs
        ):
            if token_str:
                generated_text += token_str
                print(json.dumps({'token': token_str}), flush=True)
                
                # Stop 토큰 체크
                if stop_tokens:
                    for stop_token in stop_tokens:
                        if stop_token in generated_text:
                            print(json.dumps({'stop': True}), flush=True)
                            return
        
        print(json.dumps({'stop': True}), flush=True)
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\\n{traceback.format_exc()}"
        print(json.dumps({'error': error_msg}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
`;

    try {
      fs.writeFileSync(scriptPath, scriptContent, { mode: 0o755 });
      console.log(`[MLX] Created inference script: ${scriptPath}`);
    } catch (error) {
      console.error(`[MLX] Failed to create inference script:`, error);
      throw error;
    }
  }

  createTokenizeScript(scriptPath) {
    const scriptContent = `#!/usr/bin/env python3
import json
import sys
import argparse
from mlx_lm import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', required=True)
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load(args.model)
        tokens = tokenizer.encode(args.text)
        print(json.dumps({'tokens': tokens}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
`;

    fs.writeFileSync(scriptPath, scriptContent, { mode: 0o755 });
  }

  createDetokenizeScript(scriptPath) {
    const scriptContent = `#!/usr/bin/env python3
import json
import sys
import argparse
from mlx_lm import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokens', required=True)
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load(args.model)
        tokens = json.loads(args.tokens)
        text = tokenizer.decode(tokens)
        print(json.dumps({'content': text}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
`;

    fs.writeFileSync(scriptPath, scriptContent, { mode: 0o755 });
  }

  createApplyTemplateScript(scriptPath) {
    const scriptContent = `#!/usr/bin/env python3
import json
import sys
import argparse
from jinja2 import Template

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--messages', required=True)
    
    args = parser.parse_args()
    
    try:
        # 템플릿 파일 읽기
        with open(args.template, 'r') as f:
            template_str = f.read()
        
        # 메시지 파싱
        messages = json.loads(args.messages)
        
        # Jinja 템플릿 적용
        template = Template(template_str)
        prompt = template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token='<|begin_of_text|>',
            eos_token='<|end_of_text|>',
        )
        
        print(json.dumps({'prompt': prompt}), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}), flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
`;

    fs.writeFileSync(scriptPath, scriptContent, { mode: 0o755 });
  }

  async handleLogsStream(req, res) {
    // 로그 스트림 엔드포인트 (SSE)
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    // 로그 버퍼 (최근 로그 저장)
    if (!this.logBuffer) {
      this.logBuffer = [];
      this.maxLogBufferSize = 1000;
    }

    // 기존 로그 전송
    const sendLog = (text) => {
      const logData = {
        text: text,
        timestamp: new Date().toISOString()
      };
      res.write(`event: log\ndata: ${JSON.stringify(logData)}\n\n`);
    };

    // 버퍼에 있는 로그 전송
    this.logBuffer.forEach(log => {
      sendLog(log);
    });

    // 새로운 로그를 받기 위한 리스너
    const logListener = (message) => {
      if (typeof message === 'string') {
        this.logBuffer.push(message);
        if (this.logBuffer.length > this.maxLogBufferSize) {
          this.logBuffer.shift();
        }
        sendLog(message);
      }
    };

    // console.log, console.error를 가로채서 로그 수집
    const originalLog = console.log;
    const originalError = console.error;
    
    console.log = (...args) => {
      const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ');
      logListener(message);
      originalLog.apply(console, args);
    };

    console.error = (...args) => {
      const message = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ');
      logListener(message);
      originalError.apply(console, args);
    };

    // 클라이언트 연결 종료 감지
    req.on('close', () => {
      console.log = originalLog;
      console.error = originalError;
      console.log('[MLX] Log stream closed');
    });
  }

  stop() {
    return new Promise((resolve) => {
      if (this.server) {
        console.log(`[MLX] Stopping server on port ${this.port}`);
        this.server.close(() => {
          console.log(`[MLX] Server stopped`);
          this.server = null;
          this.isModelLoaded = false;
          this.modelMeta = null;
          resolve();
        });
      } else {
        resolve();
      }
    });
  }
}

module.exports = MlxServer;
