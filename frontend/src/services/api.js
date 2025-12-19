export const LLAMA_BASE_URL = import.meta.env.VITE_LLAMACPP_BASE_URL || 'http://localhost:8080';

// 모델 형식에 따라 적절한 포트 선택
export const getServerUrl = (modelFormat = 'gguf') => {
  const baseUrl = LLAMA_BASE_URL.replace(':8080', '');
  // GGUF: 8080, MLX: 8081
  const port = modelFormat === 'mlx' ? 8081 : 8080;
  return `${baseUrl}:${port}`;
};

// 현재 활성 모델의 형식을 가져오기
export const getActiveModelFormat = () => {
  try {
    // llmServerClientConfig를 먼저 확인 (여러 모델 관리용)
    const clientConfigStr = localStorage.getItem('llmServerClientConfig');
    if (clientConfigStr) {
      try {
        const config = JSON.parse(clientConfigStr);
        if (config && config.models && Array.isArray(config.models)) {
          const activeModelId = config.activeModelId;
          const activeModel = config.models.find(m => m.id === activeModelId);
          if (activeModel) {
            return activeModel.modelFormat || 'gguf';
          }
        }
      } catch (e) {
        // 파싱 실패 시 무시
      }
    }
    
    // modelConfig 확인 (단일 모델 설정, 하위 호환성)
    const modelConfigStr = localStorage.getItem('modelConfig');
    if (modelConfigStr) {
      try {
        const modelConfig = JSON.parse(modelConfigStr);
        if (modelConfig && modelConfig.modelFormat) {
          return modelConfig.modelFormat;
        }
      } catch (e) {
        // 파싱 실패 시 무시
      }
    }
    
    // 기본값: GGUF
    return 'gguf';
  } catch (e) {
    return 'gguf'; // 기본값
  }
};

// 현재 활성 모델의 형식을 가져와서 서버 URL 반환
export const getActiveServerUrl = () => {
  const modelFormat = getActiveModelFormat();
  return getServerUrl(modelFormat);
};

// Server Log 패널에 메시지를 보내기 위한 헬퍼
const pushServerLog = (message, data) => {
  try {
    const details = data ? `${message} ${JSON.stringify(data)}` : message;
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('server-log', { detail: details }));
    }
  } catch (e) {
    // 로깅 중 오류는 무시
  }
};

export const buildLlama3Prompt = (messages, language = 'ko') => {
  // 언어별 시스템 프롬프트 설정
  const systemPrompts = {
    ko: [
      "당신은 유용한 AI 어시스턴트 '뤼(Luu)'입니다.",
      "사용자의 질문에 한국어로 답변하세요.",
      "사용자에게는 항상 존댓말을 사용하여 공손하고 예의 바르게 말하세요.",
      "답변은 반드시 3~5문장 이내로 명확하게 끝내세요."
    ].join(" "),
    en: [
      "You are a helpful AI assistant.",
      "You must respond **only in English**.",
      "Even if the user asks in another language, the final output must always be natural English.",
      "If you start generating text in another language, immediately switch back and answer in English.",
      "Keep your answers concise and focused on the key points."
    ].join(" ")
  };
  
  const systemPrompt = systemPrompts[language] || systemPrompts['ko'];
  let prompt = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${systemPrompt}<|eot_id|>`;

  messages.forEach(message => {
    const role = message.role === 'assistant' ? 'assistant' : 'user';
    prompt += `<|start_header_id|>${role}<|end_header_id|>\n\n${message.content}<|eot_id|>`;
  });

  prompt += `<|start_header_id|>assistant<|end_header_id|>\n\n`;
  return prompt;
};

export const sendChatMessage = async (messages, onToken, language = 'ko', showSpecialTokens = false) => {
  try {
    const prompt = buildLlama3Prompt(messages, language);
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    const contextSize = config.contextSize || 2048;
    const model = (config.modelPath || '').trim();
    
    // 모델 형식 확인
    const modelFormat = getActiveModelFormat();
    const useWebSocket = modelFormat === 'mlx';
    
    // 서버의 tokenize API를 사용하여 정확한 토큰 수 계산
    const promptTokens = await countTokens(prompt);
    const promptLength = prompt.length;
    
    // console.log(`[API] Prompt length: ${promptLength} chars, Actual tokens: ${promptTokens}, Context size: ${contextSize}`);
    // console.log(`[API] Token ratio: ${(promptLength / promptTokens).toFixed(2)} chars/token`);
    pushServerLog('[API] Prompt tokens info', {
      promptLength,
      promptTokens,
      contextSize,
      ratio: Number((promptLength / promptTokens).toFixed(2)),
    });
    
    if (promptTokens > contextSize) {
      // console.error(`[API] Error: Actual tokens (${promptTokens}) exceeds context size (${contextSize})`);
      pushServerLog('[API][STOP-DEBUG] Prompt exceeds context size', {
        promptTokens,
        contextSize,
      });
      throw new Error(`프롬프트가 컨텍스트 크기(${contextSize})를 초과합니다. (사용: ${promptTokens} 토큰) 대화를 초기화하거나 컨텍스트 크기를 늘려주세요.`);
    } else if (promptTokens > contextSize * 0.9) {
      // console.warn(`[API] Warning: Actual tokens (${promptTokens}) is close to context size (${contextSize})`);
      pushServerLog('[API][STOP-DEBUG] Prompt tokens close to context limit', {
        promptTokens,
        contextSize,
      });
    }
    
    // 동적 n_predict 계산: 설정값, 남은 컨텍스트, 프롬프트 길이를 모두 고려
    const maxTokensConfig = config.maxTokens || 1024;
    // -1 또는 0이면 무제한 모드 (컨텍스트 한도 내)
    let dynamicNPredict = maxTokensConfig;
    
    if (maxTokensConfig > 0) {
      const remainingContext = Math.max(contextSize - promptTokens, 0);
      // 프롬프트 토큰의 최대 4배까지만 생성하도록 제한 (불필요한 장문 방지)
      const maxByPrompt = Math.max(promptTokens * 4, 64); // 너무 작아지지 않도록 최소 64
      // 컨텍스트의 80%까지만 생성에 사용 (안전 마진)
      const maxByContext = remainingContext > 0 ? Math.floor(remainingContext * 0.8) : maxTokensConfig;
      dynamicNPredict = Math.min(maxTokensConfig, maxByPrompt, maxByContext);
      // 최소 32 토큰은 허용
      dynamicNPredict = Math.max(32, dynamicNPredict);
      
      // console.log(`[API] Dynamic n_predict: ${dynamicNPredict} (configMax=${maxTokensConfig}, remainingContext=${remainingContext}, maxByPrompt=${maxByPrompt}, maxByContext=${maxByContext})`);
      pushServerLog('[API] Dynamic n_predict calculated', {
        dynamicNPredict,
        configMax: maxTokensConfig,
        remainingContext,
        maxByPrompt,
        maxByContext,
      });
    } else {
       // 무제한 모드 로그
       // console.log(`[API] Unlimited generation mode (n_predict=${maxTokensConfig}) - STOP via EOS/Context only`);
       pushServerLog('[API] Unlimited generation mode', {
         configMax: maxTokensConfig,
         contextSize,
         note: 'Only stops at EOS or Context Limit'
       });
       dynamicNPredict = -1;
    }
    
    const serverUrl = getActiveServerUrl();
    
    // MLX 모델은 WebSocket 사용
    if (useWebSocket && typeof WebSocket !== 'undefined') {
      return new Promise((resolve, reject) => {
        const wsUrl = serverUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        const ws = new WebSocket(`${wsUrl}/chat/ws`);
        
        ws.onopen = () => {
          ws.send(JSON.stringify({
            prompt: prompt,
            max_tokens: dynamicNPredict > 0 ? dynamicNPredict : 2048,
            temperature: config.temperature ?? 0.7,
            top_p: config.topP || 0.95,
            min_p: config.minP || 0.05,
            repeat_penalty: config.repeatPenalty || 1.1,
            repeat_last_n: config.repeatLastN || 64,
          }));
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'token' && data.content) {
              let token = data.content;
              // 스페셜 토큰 표시가 꺼져있으면 스페셜 토큰 제거
              if (!showSpecialTokens) {
                token = token.replace(/<\|[^>]*\|>/g, '');
              }
              if (token) {
                onToken(token);
                // token-received 이벤트 발생 (PerformancePanel에서 토큰 속도 계산용)
                window.dispatchEvent(new CustomEvent('token-received'));
                // 디버깅용
                // console.log('[API] token-received event dispatched for token:', token.substring(0, 20));
              }
            } else if (data.type === 'done') {
              ws.close();
              resolve();
            } else if (data.type === 'error') {
              ws.close();
              reject(new Error(data.message || 'WebSocket error'));
            }
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e);
          }
        };
        
        ws.onerror = (error) => {
          ws.close();
          reject(new Error('WebSocket connection error'));
        };
        
        ws.onclose = () => {
          resolve();
        };
      });
    }
    
    // GGUF 모델은 기존 HTTP POST + SSE 방식 사용
    // llama.cpp server uses snake_case for parameters
    const payload = {
      model,
      prompt,
      stream: true,
      n_predict: dynamicNPredict,
      temperature: config.temperature ?? 0.7,
      top_k: config.topK || 40,
      top_p: config.topP || 0.95,
      min_p: config.minP || 0.05,
      tfs_z: config.tfsZ || 1.0,
      typical_p: config.typicalP || 1.0,
      repeat_penalty: config.repeatPenalty || 1.1,
      repeat_last_n: config.repeatLastN || 64,
      penalize_nl: config.penalizeNL || false,
      dry_multiplier: config.dryMultiplier || 0.0,
      dry_base: config.dryBase || 1.75,
      dry_allowed_length: config.dryAllowedLength || 2,
      dry_penalty_last_n: config.dryPenaltyLastN || -1,
      mirostat: config.mirostatMode || 0,
      mirostat_tau: config.mirostatTau || 5.0,
      mirostat_eta: config.mirostatEta || 0.1,
      // 스페셜 토큰 표시가 ON이면 stop 파라미터를 비워서 스페셜 토큰이 중단되지 않도록 함
      stop: showSpecialTokens ? [] : ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "~HAPY~", "~~", "!!", "..", "ㅋㅋ", "ㅎㅎ", "\n\n"],
      // 스페셜 토큰을 반환하도록 설정 (가능한 경우)
      return_tokens: showSpecialTokens
    };

    // console.log('[API] Request Payload:', JSON.stringify(payload, null, 2)); // 디버그용 Payload 로그 추가

    const response = await fetch(`${serverUrl}/completion`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      // console.error("[API] Server error:", errorText);
      // console.error("[API] Request payload size:", JSON.stringify(payload).length, "bytes");
      // console.error("[API] Prompt size:", prompt.length, "chars");
      // console.error("[API] Actual tokens:", promptTokens);
      pushServerLog('[API] Server error response', {
        status: response.status,
        errorText,
        requestPayloadBytes: JSON.stringify(payload).length,
        promptChars: prompt.length,
        promptTokens,
      });
      
      // 503 에러 (모델 로딩 중) 감지
      if (response.status === 503) {
        window.dispatchEvent(new CustomEvent('model-loading', { detail: { loading: true } }));
      }
      
      // 에러 응답에서 실제 토큰 수 파싱 시도
      try {
        const errorJson = JSON.parse(errorText);
        if (errorJson.error && errorJson.error.n_prompt_tokens) {
          const actualTokens = errorJson.error.n_prompt_tokens;
          // console.error(`[API] Server reported actual tokens: ${actualTokens}`);
          // 실제 토큰 수를 이벤트로 전달하여 UI 업데이트
          window.dispatchEvent(new CustomEvent('context-update', {
            detail: {
              used: actualTokens,
              total: contextSize
            }
          }));
        }
      } catch (e) {
        // JSON 파싱 실패 시 무시
      }
      
      throw new Error(`Server responded with status: ${response.status}. ${errorText}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let partialData = '';
    let tokenStartTime = Date.now();
    let tokenCount = 0;
    let stoppedByServer = false;
    let lastParsedChunk = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      
      partialData += decoder.decode(value, { stream: true });

      // The server sends data in "data: { ...json... }" format.
      // We need to parse these chunks.
      const lines = partialData.split('\n');
      partialData = lines.pop(); // Keep the last, possibly incomplete, line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonString = line.substring(6);
          if (jsonString) {
            try {
              const parsed = JSON.parse(jsonString);
              lastParsedChunk = parsed;
              if (parsed.content) {
                let token = parsed.content;
                // 스페셜 토큰 표시가 꺼져있으면 스페셜 토큰 제거
                if (!showSpecialTokens) {
                  // 스페셜 토큰 패턴 제거: <|...|> (|>로 끝나는 모든 패턴)
                  token = token.replace(/<\|[^>]*\|>/g, '');
                }
                // 스페셜 토큰 표시가 ON이면 모든 스페셜 토큰을 그대로 전달
                // 스페셜 토큰이 있으면 그대로 전달
                if (token) {
                  onToken(token);
                  tokenCount++;
                  // 토큰 수신 이벤트 발생 (Performance 패널에서 사용)
                  window.dispatchEvent(new CustomEvent('token-received'));
                }
              }
              // return_tokens가 true일 때 토큰 ID도 확인
              if (showSpecialTokens && parsed.tokens && Array.isArray(parsed.tokens) && parsed.tokens.length > 0) {
                // 토큰 ID가 있으면 이를 통해 스페셜 토큰을 확인할 수 있음
                // 하지만 일반적으로 content에 이미 포함되어 있음
              }

              // 서버가 시퀀스를 잘랐다고 보고하는 경우 (컨텍스트 초과 등)
              if (parsed.truncated) {
                // console.warn('[API][STOP-DEBUG] Server reports truncated sequence:', parsed);
                pushServerLog('[API][STOP-DEBUG] Server reports truncated sequence', {
                  truncated: parsed.truncated,
                  stop: parsed.stop,
                  stop_reason: parsed.stop_reason ?? parsed.stop_type,
                  tokenCount,
                  n_predict: payload.n_predict,
                  contextSize,
                });
              }

              // 서버에서 명시적으로 stop 플래그를 보낸 경우
              if (parsed.stop) {
                const duration = Date.now() - tokenStartTime;
                const stopInfo = {
                  stop: parsed.stop,
                  truncated: parsed.truncated,
                  stop_reason: parsed.stop_reason ?? parsed.stop_type,
                  n_predict: payload.n_predict,
                  contextSize,
                  stopArray: payload.stop,
                  showSpecialTokens,
                  tokenCount,
                  durationMs: duration,
                };
                // console.log('[API][STOP-DEBUG] Server sent stop signal:', stopInfo);
                pushServerLog('[API][STOP-DEBUG] Server sent stop signal', stopInfo);
                // console.log(`[API] Generation completed: ${tokenCount} tokens in ${duration}ms`);
                stoppedByServer = true;
                return; // Stop processing further tokens
              }
            } catch (e) {
              console.error('Failed to parse stream chunk:', jsonString, e);
            }
          }
        }
      }
    }

    // 스트림이 done으로 끝났는데 서버의 stop 플래그가 없었던 경우
    if (!stoppedByServer) {
      const totalDuration = Date.now() - tokenStartTime;
      const endInfo = {
        tokenCount,
        totalDurationMs: totalDuration,
        lastChunk: lastParsedChunk,
        n_predict: payload.n_predict,
        contextSize,
        stopArray: payload.stop,
        showSpecialTokens,
      };
      // console.warn('[API][STOP-DEBUG] Stream ended without explicit stop flag from server.', endInfo);
      pushServerLog('[API][STOP-DEBUG] Stream ended without explicit stop flag from server', endInfo);
    }

  } catch (error) {
    console.error('채팅 스트림 오류:', error);
    throw error;
  }
};

// 정확한 토큰 수 계산을 위한 함수
export const countTokens = async (prompt) => {
  try {
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    const model = (config.modelPath || '').trim();
    const serverUrl = getActiveServerUrl();
    const response = await fetch(`${serverUrl}/tokenize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        content: prompt,
        add_special: false,
        parse_special: true
      }),
      signal: AbortSignal.timeout(2000), // 2초 타임아웃
    });

    if (!response.ok) {
      // tokenize API가 실패하면 추정값 사용 (조용히 처리)
      return Math.ceil(prompt.length / 2.0);
    }

    const data = await response.json();
    if (data.tokens && Array.isArray(data.tokens)) {
      return data.tokens.length;
    }
    
    // 응답 형식이 다를 경우 추정값 사용
    return Math.ceil(prompt.length / 2.0);
  } catch (error) {
    // API 호출 실패 시 보수적인 추정값 사용 (조용히 처리)
    // AbortError나 네트워크 에러는 로그 출력 안 함
    if (error.name !== 'AbortError' && !error.message.includes('Failed to fetch')) {
      // console.warn('[API] Tokenize API error, using estimation:', error);
    }
    return Math.ceil(prompt.length / 2.0);
  }
};

// 토큰 디버깅용 함수: 토큰 ID 및 piece 반환
export const tokenizeText = async (content) => {
  try {
    if (!content || content.trim() === '') {
      console.warn('[API] tokenizeText: Empty content');
      return [];
    }
    
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    const model = (config.modelPath || '').trim();
    const serverUrl = getActiveServerUrl();
    
    // console.log('[API] tokenizeText: Calling', `${serverUrl}/tokenize`, 'with content length:', content.length);
    
    const response = await fetch(`${serverUrl}/tokenize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        content,
        add_special: true,
        parse_special: true,
        with_pieces: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.warn('[API] Tokenize API (with_pieces) failed:', response.status, errorText);
      return [];
    }

    const data = await response.json();
    // console.log('[API] tokenizeText: Response received:', data.tokens?.length || 0, 'tokens', 'data:', data);
    if (data.tokens && Array.isArray(data.tokens)) {
      return data.tokens;
    }

    console.warn('[API] Unexpected tokenize (with_pieces) response format:', data);
    return [];
  } catch (error) {
    console.warn('[API] Tokenize (with_pieces) error:', error);
    return [];
  }
};

// llama-server는 /system 같은 별도 시스템 엔드포인트를 제공하지 않습니다.
// 필요 시 /props 또는 /metrics를 사용하세요.
