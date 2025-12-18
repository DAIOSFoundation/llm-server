/**
 * MLX Python 기반 서버
 * Python mlx_lm 엔진과 IPC로 통신하여 추론을 수행합니다.
 */
const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();

// CORS 미들웨어
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-llm-ui-auth');
    
    // OPTIONS 요청에 대한 응답
    if (req.method === 'OPTIONS') {
        res.sendStatus(200);
        return;
    }
    
    next();
});

app.use(express.json());

// --- AI 엔진 클래스 ---
class AIEngine {
    constructor() {
        this.process = null;
        this.ready = false;
        this.currentRes = null; // 현재 응답 중인 Response 객체
        this.requestQueue = []; // 요청 큐
        this.processing = false; // 현재 처리 중인지 여부
        this.logListeners = []; // 로그 스트림 리스너들
        this.metricsListeners = []; // 메트릭 스트림 리스너들
        this.tokenCount = 0; // 생성된 토큰 수
        this.startTime = null; // 요청 시작 시간
        this.lastTokenTime = null; // 마지막 토큰 생성 시간
    }

    start() {
        const enginePath = path.join(__dirname, 'engine.py');
        
        // Python 파일 존재 확인
        if (!fs.existsSync(enginePath)) {
            console.error(`[ERROR] engine.py not found at ${enginePath}`);
            return;
        }

        // Python 프로세스 실행
        // 가상환경을 쓴다면: 'venv/bin/python3'
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        this.process = spawn(pythonCmd, [enginePath], {
            cwd: __dirname,
            env: {
                ...process.env,
                MLX_MODEL_PATH: process.env.MLX_MODEL_PATH || './models/deepseek-moe-16b-chat-mlx-q4_0'
            }
        });

        // Python -> Node.js 메시지 수신
        let buffer = '';
        this.process.stdout.on('data', (data) => {
            const output = data.toString();
            buffer += output;
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 보관
            
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const msg = JSON.parse(line);
                    this.handleMessage(msg);
                } catch (e) {
                    // JSON이 아닌 일반 로그 메시지인 경우
                    const logMessage = `[Engine] ${line}`;
                    console.log(logMessage);
                    this.broadcastLog(logMessage);
                }
            }
        });

        this.process.stderr.on('data', (data) => {
            const logMessage = data.toString();
            console.error(`[Engine Log] ${logMessage}`);
            // 로그 스트림 리스너들에게 브로드캐스트
            this.broadcastLog(logMessage.trim());
        });

        this.process.on('close', (code) => {
            const logMessage = `[Engine] AI Engine process exited with code ${code}`;
            console.log(logMessage);
            this.broadcastLog(logMessage);
            this.ready = false;
            // 프로세스가 종료되면 재시작 시도
            if (code !== 0) {
                const restartMessage = '[Engine] Restarting in 3 seconds...';
                console.log(restartMessage);
                this.broadcastLog(restartMessage);
                setTimeout(() => this.start(), 3000);
            }
        });

        this.process.on('error', (err) => {
            const errorMessage = `[Engine] Failed to start Python process: ${err.message}`;
            console.error(errorMessage);
            this.broadcastLog(errorMessage);
            if (err.code === 'ENOENT') {
                const installMessage = '[Engine] Python3 not found. Please install Python 3.8+ and mlx-lm: pip install mlx-lm';
                console.error(installMessage);
                this.broadcastLog(installMessage);
            }
        });
    }

    handleMessage(msg) {
        if (msg.status === 'ready') {
            const logMessage = '✅ AI Engine is Ready!';
            console.log(logMessage);
            this.broadcastLog(logMessage);
            this.ready = true;
            this.processing = false;
            this.processNextRequest();
        } 
        else if (msg.status === 'token') {
            // SSE 형식으로 클라이언트에 전송
            if (this.currentRes) {
                // Content-Type이 이미 설정되어 있으면 그대로 사용, 아니면 SSE 형식으로 설정
                if (!this.currentRes.headersSent) {
                    const contentType = this.currentRes.getHeader('Content-Type');
                    if (!contentType || !contentType.includes('event-stream')) {
                        // SSE 형식으로 설정
                        this.currentRes.setHeader('Content-Type', 'text/event-stream');
                        this.currentRes.setHeader('Cache-Control', 'no-cache');
                        this.currentRes.setHeader('Connection', 'keep-alive');
                    }
                }
                // SSE 형식으로 전송
                this.currentRes.write(`data: ${JSON.stringify({ content: msg.content })}\n\n`);
            }
            // 메트릭 업데이트: 토큰 카운트 증가
            this.tokenCount++;
            this.lastTokenTime = Date.now();
        } 
        else if (msg.status === 'done') {
            if (this.currentRes) {
                // SSE 형식으로 종료 신호 전송
                this.currentRes.write(`data: ${JSON.stringify({ stop: true })}\n\n`);
                this.currentRes.end();
                this.currentRes = null;
            }
            // 메트릭 업데이트: 처리 완료
            const duration = this.startTime ? (Date.now() - this.startTime) / 1000 : 0;
            const tokensPerSecond = duration > 0 ? (this.tokenCount / duration).toFixed(2) : 0;
            this.broadcastLog(`[Engine] Generation completed: ${this.tokenCount} tokens in ${duration.toFixed(2)}s (${tokensPerSecond} tokens/s)`);
            this.processing = false;
            this.tokenCount = 0;
            this.startTime = null;
            this.lastTokenTime = null;
            this.processNextRequest();
        }
        else if (msg.status === 'error') {
            const errorMessage = `[Engine] AI Engine Error: ${msg.message}`;
            console.error(errorMessage);
            this.broadcastLog(errorMessage);
            if (this.currentRes) {
                // Transfer-Encoding 헤더 제거 후 JSON 응답
                const res = this.currentRes;
                // 헤더 완전히 정리
                res.removeHeader('Transfer-Encoding');
                res.removeHeader('Content-Length');
                res.removeHeader('Content-Type');
                // 수동으로 JSON 응답 작성 (Express의 json()은 Content-Length를 자동 설정)
                const errorResponse = JSON.stringify({ error: msg.message });
                res.status(500);
                res.setHeader('Content-Type', 'application/json');
                res.setHeader('Content-Length', Buffer.byteLength(errorResponse));
                res.end(errorResponse);
                this.currentRes = null;
            }
            this.processing = false;
            this.processNextRequest();
        }
        else if (msg.status === 'loading') {
            const logMessage = `[Engine] ${msg.message}`;
            console.log(logMessage);
            this.broadcastLog(logMessage);
        }
        else if (msg.status === 'warning') {
            const warningMessage = `[Engine] ⚠️ ${msg.message}`;
            console.warn(warningMessage);
            this.broadcastLog(warningMessage);
        }
        else if (msg.status === 'ready') {
            const logMessage = `[Engine] ${msg.message}`;
            console.log(logMessage);
            this.broadcastLog(logMessage);
        }
    }

    processNextRequest() {
        if (this.processing || !this.ready || this.requestQueue.length === 0) {
            return;
        }

        const { prompt, res, options } = this.requestQueue.shift();
        this.generate(prompt, res, options);
    }

    generate(prompt, res, options = {}) {
        if (!this.ready) {
            res.removeHeader('Transfer-Encoding');
            res.removeHeader('Content-Length');
            const errorResponse = JSON.stringify({ error: 'Model is loading...' });
            res.status(503);
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Length', Buffer.byteLength(errorResponse));
            res.end(errorResponse);
            return;
        }
        
        // 큐에 추가
        if (this.processing) {
            this.requestQueue.push({ prompt, res, options });
            return;
        }

        this.processing = true;
        this.currentRes = res;
        this.tokenCount = 0;
        this.startTime = Date.now();
        this.lastTokenTime = null;
        
        // SSE 형식으로 스트리밍 Response 설정
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Transfer-Encoding', 'chunked');
        res.removeHeader('Content-Length'); // Content-Length 제거 (Transfer-Encoding과 충돌 방지)
        
        // Python에 요청 전송
        // mlx_lm이 지원하는 파라미터만 전송
        const request = {
            prompt: prompt,
            max_tokens: options.max_tokens || options.n_predict || 1024,
            temperature: options.temperature || 0.7,
            top_p: options.top_p || 0.95,
            min_p: options.min_p || 0.0,
            repeat_penalty: options.repeat_penalty || 1.1,
            repeat_last_n: options.repeat_last_n || 64,
            // mlx_lm에서 지원하지 않는 파라미터들 (무시되지만 전송하여 경고 메시지 확인 가능)
            top_k: options.top_k || options.topK,  // 지원하지 않음 (경고용)
            mirostat: options.mirostat || options.mirostatMode,
            tfs_z: options.tfs_z || options.tfsZ,
            typical_p: options.typical_p || options.typicalP,
            penalize_nl: options.penalize_nl || options.penalizeNL,
            dry_multiplier: options.dry_multiplier || options.dryMultiplier,
            presence_penalty: options.presence_penalty || options.presencePenalty,
            frequency_penalty: options.frequency_penalty || options.frequencyPenalty
        };
        
        try {
            this.process.stdin.write(JSON.stringify(request) + '\n');
        } catch (e) {
            console.error('[Engine] Failed to send request:', e);
            // 헤더 정리 후 수동으로 JSON 응답
            res.removeHeader('Transfer-Encoding');
            res.removeHeader('Content-Length');
            res.removeHeader('Content-Type');
            const errorResponse = JSON.stringify({ error: 'Failed to send request to engine' });
            res.status(500);
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Length', Buffer.byteLength(errorResponse));
            res.end(errorResponse);
            this.currentRes = null;
            this.processing = false;
        }
    }

    stop() {
        if (this.process) {
            this.process.kill();
            this.process = null;
        }
        this.ready = false;
        // 모든 리스너 정리
        this.logListeners = [];
        this.metricsListeners = [];
    }

    // 로그 브로드캐스트
    broadcastLog(message) {
        const logData = JSON.stringify({ text: message });
        this.logListeners.forEach(res => {
            if (!res.writableEnded) {
                try {
                    res.write(`event: log\n`);
                    res.write(`data: ${logData}\n\n`);
                } catch (e) {
                    // 연결이 끊어진 경우 리스너에서 제거
                    const index = this.logListeners.indexOf(res);
                    if (index > -1) {
                        this.logListeners.splice(index, 1);
                    }
                }
            }
        });
    }

    // 메트릭 리스너 추가
    addMetricsListener(res) {
        this.metricsListeners.push(res);
    }

    // 로그 리스너 추가
    addLogListener(res) {
        this.logListeners.push(res);
    }

    // 리스너 제거
    removeListener(res) {
        const logIndex = this.logListeners.indexOf(res);
        if (logIndex > -1) {
            this.logListeners.splice(logIndex, 1);
        }
        const metricsIndex = this.metricsListeners.indexOf(res);
        if (metricsIndex > -1) {
            this.metricsListeners.splice(metricsIndex, 1);
        }
    }
}

// --- 서버 실행 ---
const engine = new AIEngine();
engine.start();

// Health check
app.get('/health', (req, res) => {
    res.json({ 
        status: engine.ready ? 'ready' : 'loading',
        engine: 'python-mlx-lm'
    });
});

// Metrics
app.get('/metrics', (req, res) => {
    res.json({
        ready: engine.ready,
        processing: engine.processing,
        queueLength: engine.requestQueue.length,
        engine: 'python-mlx-lm'
    });
});

// Chat endpoint
app.post('/chat', (req, res) => {
    const { prompt, ...options } = req.body;
    
    if (!prompt) {
        res.status(400).json({ error: 'Prompt is required' });
        return;
    }
    
    engine.generate(prompt, res, options);
});

// Completion endpoint (llama.cpp 호환)
app.post('/completion', (req, res) => {
    const { prompt, stream = true, ...options } = req.body;
    
    if (!prompt) {
        res.status(400).json({ error: 'Prompt is required' });
        return;
    }
    
    if (stream) {
        // 스트리밍 응답 (SSE 형식)
        engine.generate(prompt, res, options);
    } else {
        // 비스트리밍 응답 (전체 응답을 한 번에 반환)
        res.setHeader('Content-Type', 'application/json');
        
        let fullContent = '';
        const mockRes = {
            write: (chunk) => {
                // SSE 형식에서 content 추출
                try {
                    const lines = chunk.toString().split('\n');
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.substring(6));
                            if (data.content) {
                                fullContent += data.content;
                            }
                        }
                    }
                } catch (e) {
                    fullContent += chunk.toString();
                }
            },
            end: () => {
                res.json({
                    content: fullContent,
                    stop: true
                });
            },
            setHeader: () => {},
            removeHeader: () => {},
            getHeader: () => null,
            headersSent: false,
            status: () => mockRes
        };
        
        engine.generate(prompt, mockRes, options);
    }
});

// Tokenize endpoint (llama.cpp 호환)
app.post('/tokenize', async (req, res) => {
    const { content, model } = req.body;
    
    if (!content) {
        res.status(400).json({ error: 'Content is required' });
        return;
    }
    
    // Python 엔진에 토큰화 요청 전송
    // 간단한 구현: Python 엔진이 토큰화를 지원하지 않으므로 추정값 반환
    // 실제로는 Python 엔진에 토큰화 기능을 추가해야 함
    try {
        // 임시로 문자 수 기반 추정
        const estimatedTokens = Math.ceil(content.length / 2.0);
        const tokens = Array.from({ length: estimatedTokens }, (_, i) => i);
        
        res.json({
            tokens: tokens,
            count: tokens.length
        });
    } catch (error) {
        console.error('[Server] Tokenize error:', error);
        res.status(500).json({ error: 'Tokenization failed' });
    }
});

// Metrics stream endpoint (프론트엔드 호환)
app.get('/metrics/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    // 메트릭 리스너에 추가
    engine.addMetricsListener(res);
    
    // 상세한 메트릭 전송
    const sendMetrics = () => {
        if (res.writableEnded) {
            engine.removeListener(res);
            return;
        }
        
        try {
            // 토큰 생성 속도 계산
            let tokensPerSecond = 0;
            if (engine.lastTokenTime && engine.startTime) {
                const elapsed = (Date.now() - engine.startTime) / 1000;
                if (elapsed > 0) {
                    tokensPerSecond = (engine.tokenCount / elapsed).toFixed(2);
                }
            }
            
            res.write(`event: metrics\n`);
            res.write(`data: ${JSON.stringify({
                ready: engine.ready,
                processing: engine.processing,
                queueLength: engine.requestQueue.length,
                engine: 'python-mlx-lm',
                tokensGenerated: engine.tokenCount,
                tokensPerSecond: parseFloat(tokensPerSecond) || 0,
                processingTime: engine.startTime ? ((Date.now() - engine.startTime) / 1000).toFixed(2) : 0
            })}\n\n`);
        } catch (e) {
            // 연결 오류 시 리스너 제거
            engine.removeListener(res);
        }
    };
    
    // 초기 메트릭 전송
    sendMetrics();
    
    // 주기적으로 메트릭 전송 (1초마다)
    const interval = setInterval(() => {
        if (res.writableEnded) {
            clearInterval(interval);
            engine.removeListener(res);
            return;
        }
        sendMetrics();
    }, 1000);
    
    // 클라이언트 연결 종료 시 정리
    req.on('close', () => {
        clearInterval(interval);
        engine.removeListener(res);
    });
});

// Logs stream endpoint (프론트엔드 호환)
app.get('/logs/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    
    // 로그 리스너에 추가
    engine.addLogListener(res);
    
    // 초기 로그 메시지 전송
    res.write(`event: log\n`);
    res.write(`data: ${JSON.stringify({ text: '[Server] Connected to log stream' })}\n\n`);
    
    // 연결 유지 (keep-alive)
    const keepAlive = setInterval(() => {
        if (res.writableEnded) {
            clearInterval(keepAlive);
            engine.removeListener(res);
            return;
        }
    }, 30000); // 30초마다 keep-alive
    
    // 클라이언트 연결 종료 시 정리
    req.on('close', () => {
        clearInterval(keepAlive);
        engine.removeListener(res);
    });
    
    req.on('error', () => {
        clearInterval(keepAlive);
        engine.removeListener(res);
    });
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('[Server] SIGTERM received, shutting down gracefully...');
    engine.stop();
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('[Server] SIGINT received, shutting down gracefully...');
    engine.stop();
    process.exit(0);
});

const PORT = process.env.PORT || 8081;
app.listen(PORT, () => {
    const startMessage = `[Server] MLX Python-based server started on port ${PORT}`;
    const modelPathMessage = `[Server] Model path: ${process.env.MLX_MODEL_PATH || './models/deepseek-moe-16b-chat-mlx-q4_0'}`;
    console.log(startMessage);
    console.log(modelPathMessage);
    // 로그 스트림이 아직 설정되지 않았을 수 있으므로 약간의 지연 후 전송
    setTimeout(() => {
        engine.broadcastLog(startMessage);
        engine.broadcastLog(modelPathMessage);
    }, 100);
});

