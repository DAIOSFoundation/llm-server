#!/usr/bin/env node

const http = require('http');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

// EventSource는 사용하지 않음 (HTTP 스트림 직접 파싱)

// 설정
const SERVER_PORT = 8081;
const SERVER_URL = `http://localhost:${SERVER_PORT}`;
const TEST_PROMPT = 'What is the capital of South Korea?';
const MAX_RETRIES = 1; // 테스트 1번만 실행
const HEALTH_CHECK_RETRIES = 3;
const HEALTH_CHECK_DELAY = 1000; // 1초
const SERVER_START_TIMEOUT = 30000; // 30초
const INFERENCE_TIMEOUT = 60000; // 60초

// 전역 변수
let serverProcess = null;
let testCount = 0;
let successCount = 0;
let failureCount = 0;
let cppFileWatcher = null;
let lastCppFileMtime = null;

// 로깅 유틸리티
function log(message, type = 'info') {
  const timestamp = new Date().toISOString();
  const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'warn' ? '⚠️' : 'ℹ️';
  console.log(`[${timestamp}] ${prefix} ${message}`);
}

function logError(message, error) {
  log(message, 'error');
  if (error) {
    console.error('Error details:', error);
    if (error.stack) {
      console.error('Stack trace:', error.stack);
    }
  }
}

// 서버 프로세스 종료
function stopServer() {
  return new Promise((resolve) => {
    if (serverProcess) {
      log('서버 프로세스 종료 중...');
      serverProcess.kill('SIGTERM');
      
      const timeout = setTimeout(() => {
        if (serverProcess && !serverProcess.killed) {
          log('강제 종료 중...', 'warn');
          serverProcess.kill('SIGKILL');
        }
        serverProcess = null;
        resolve();
      }, 5000);

      serverProcess.on('exit', () => {
        clearTimeout(timeout);
        serverProcess = null;
        log('서버 프로세스 종료됨');
        resolve();
      });
    } else {
      resolve();
    }
  });
}

// 서버 시작
async function startServer() {
  return new Promise(async (resolve, reject) => {
    try {
      // 기존 서버 종료
      await stopServer();

      // config.json에서 MLX 모델 찾기
      const configPath = path.join(__dirname, '..', 'config.json');
      if (!fs.existsSync(configPath)) {
        throw new Error('config.json을 찾을 수 없습니다');
      }

      const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
      const mlxModel = config.models.find(m => m.modelFormat === 'mlx');
      
      if (!mlxModel) {
        throw new Error('MLX 모델을 config.json에서 찾을 수 없습니다');
      }

      log(`MLX 모델 발견: ${mlxModel.modelPath}`);

      // 서버 시작 스크립트 생성
      const serverScript = `
const MlxServer = require('./server');
const modelConfig = ${JSON.stringify(mlxModel, null, 2)};

const server = new MlxServer(modelConfig);
server.start().then(() => {
  console.log('[Test] Server started successfully');
}).catch((error) => {
  console.error('[Test] Server start failed:', error);
  process.exit(1);
});

// 프로세스 종료 시 정리
process.on('SIGTERM', () => {
  server.stop().then(() => {
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  server.stop().then(() => {
    process.exit(0);
  });
});
`;

      const scriptPath = path.join(__dirname, 'test-server-temp.js');
      fs.writeFileSync(scriptPath, serverScript, 'utf-8');
      
      // 파일이 제대로 생성되었는지 확인
      if (!fs.existsSync(scriptPath)) {
        throw new Error('서버 스크립트 파일 생성 실패');
      }
      
      // 파일 권한 확인 및 설정
      fs.chmodSync(scriptPath, 0o755);

      // Node.js로 서버 시작
      log('서버 시작 중...');
      serverProcess = spawn('node', [scriptPath], {
        cwd: __dirname,
        stdio: 'pipe',
        env: { ...process.env }
      });

      let serverOutput = '';
      serverProcess.stdout.on('data', (data) => {
        const output = data.toString();
        serverOutput += output;
        process.stdout.write(`[Server] ${output}`);
      });

      serverProcess.stderr.on('data', (data) => {
        const output = data.toString();
        serverOutput += output;
        process.stderr.write(`[Server Error] ${output}`);
      });

      serverProcess.on('exit', (code, signal) => {
        if (code !== 0 && code !== null) {
          logError(`서버가 종료되었습니다 (코드: ${code}, signal: ${signal})`);
          log('서버 출력:', 'info');
          console.log(serverOutput);
          // 서버가 크래시된 경우 reject
          if (!serverOutput.includes('Server started on port')) {
            reject(new Error(`서버가 시작 전에 종료되었습니다 (코드: ${code})`));
          }
        }
      });
      
      // 서버 프로세스 오류 처리
      serverProcess.on('error', (error) => {
        logError('서버 프로세스 오류:', error);
        reject(error);
      });

      // 서버 시작 대기
      const startTime = Date.now();
      let serverStarted = false;
      const checkInterval = setInterval(async () => {
        // 서버 프로세스가 종료되었는지 확인
        if (serverProcess && serverProcess.killed) {
          clearInterval(checkInterval);
          if (!serverStarted) {
            reject(new Error('서버 프로세스가 시작 전에 종료되었습니다'));
          }
          return;
        }
        
        // 서버 출력에서 "Server started" 메시지 확인
        if (serverOutput.includes('Server started on port') || serverOutput.includes('[MLX] Server started on port')) {
          serverStarted = true;
        }
        
        try {
          const response = await new Promise((resolve, reject) => {
            const req = http.get(`${SERVER_URL}/health`, (res) => {
              resolve({ statusCode: res.statusCode });
            });
            req.on('error', (err) => {
              // ECONNREFUSED는 서버가 아직 시작되지 않았음을 의미
              if (err.code === 'ECONNREFUSED') {
                reject(new Error('Connection refused'));
              } else {
                reject(err);
              }
            });
            req.setTimeout(2000, () => {
              req.destroy();
              reject(new Error('Health check 타임아웃'));
            });
          });
          
          if (response.statusCode === 200) {
            clearInterval(checkInterval);
            log('서버가 시작되었습니다');
            serverStarted = true;
            resolve();
          }
        } catch (error) {
          // 서버가 아직 시작되지 않음 - 정상적인 상황
          if (Date.now() - startTime > SERVER_START_TIMEOUT) {
            clearInterval(checkInterval);
            reject(new Error(`서버 시작 타임아웃 (${SERVER_START_TIMEOUT}ms). 서버 출력:\n${serverOutput}`));
          }
          // ECONNREFUSED는 정상적인 상황 (서버가 아직 시작 중)
        }
      }, 1000); // 1초마다 체크

      // 타임아웃 설정
      setTimeout(() => {
        clearInterval(checkInterval);
        if (!serverStarted && serverProcess && !serverProcess.killed) {
          logError('서버 시작 타임아웃. 서버 출력:', serverOutput);
          reject(new Error(`서버 시작 타임아웃 (${SERVER_START_TIMEOUT}ms). 서버 출력:\n${serverOutput}`));
        }
      }, SERVER_START_TIMEOUT);

    } catch (error) {
      logError('서버 시작 실패', error);
      reject(error);
    }
  });
}

// Health check 테스트
async function testHealthCheck() {
  log('Health check 테스트 시작...');
  
  for (let i = 0; i < HEALTH_CHECK_RETRIES; i++) {
    try {
      const response = await new Promise((resolve, reject) => {
        const req = http.get(`${SERVER_URL}/health`, (res) => {
          let data = '';
          res.on('data', (chunk) => { data += chunk; });
          res.on('end', () => {
            resolve({ statusCode: res.statusCode, data });
          });
        });
        req.on('error', reject);
        req.setTimeout(5000, () => {
          req.destroy();
          reject(new Error('Health check 타임아웃'));
        });
      });
      
      if (response.statusCode !== 200) {
        throw new Error(`HTTP ${response.statusCode}`);
      }

      const data = JSON.parse(response.data);
      
      if (data.status === 'ok') {
        log('Health check 성공', 'success');
        return true;
      } else {
        throw new Error(`예상하지 못한 응답: ${JSON.stringify(data)}`);
      }
    } catch (error) {
      if (i < HEALTH_CHECK_RETRIES - 1) {
        log(`Health check 실패 (재시도 ${i + 1}/${HEALTH_CHECK_RETRIES})`, 'warn');
        await new Promise(resolve => setTimeout(resolve, HEALTH_CHECK_DELAY));
      } else {
        logError('Health check 실패', error);
        throw error;
      }
    }
  }
  
  return false;
}

// Metrics 테스트
async function testMetrics() {
  log('Metrics 테스트 시작...');
  
  try {
    const response = await new Promise((resolve, reject) => {
      const req = http.get(`${SERVER_URL}/metrics`, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          resolve({ statusCode: res.statusCode, data });
        });
      });
      req.on('error', reject);
      req.setTimeout(5000, () => {
        req.destroy();
        reject(new Error('Metrics 타임아웃'));
      });
    });
    
    if (response.statusCode !== 200) {
      throw new Error(`HTTP ${response.statusCode}`);
    }

    const text = response.data;
    
    // Prometheus 형식의 메트릭 확인
    if (text.includes('vram_total_bytes') || text.includes('vram_used_bytes')) {
      log('Metrics 엔드포인트 성공', 'success');
      log(`Metrics 내용:\n${text}`, 'info');
      return true;
    } else {
      throw new Error('예상하지 못한 metrics 형식');
    }
  } catch (error) {
    logError('Metrics 테스트 실패', error);
    throw error;
  }
}

// Metrics Stream 테스트
async function testMetricsStream() {
  log('Metrics Stream 테스트 시작...');
  
  return new Promise((resolve, reject) => {
    try {
      const url = new URL(`${SERVER_URL}/metrics/stream?interval_ms=1000`);
      
      const options = {
        hostname: url.hostname,
        port: url.port || SERVER_PORT,
        path: url.pathname + url.search,
        method: 'GET',
        headers: {
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        timeout: 5000
      };

      const req = http.request(options, (res) => {
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
          return;
        }

        let buffer = '';
        let metricsReceived = false;
        let timeout;

        res.on('data', (chunk) => {
          buffer += chunk.toString();
          
          // SSE 형식 파싱: "event: metrics\ndata: {...}\n\n"
          // 빈 줄로 이벤트가 구분됨
          const events = buffer.split('\n\n');
          buffer = events.pop() || ''; // 마지막 불완전한 이벤트는 버퍼에 보관
          
          for (const event of events) {
            if (!event.trim()) continue;
            
            const lines = event.split('\n');
            let eventType = null;
            let dataLine = null;
            
            for (const line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.substring(7).trim();
              } else if (line.startsWith('data: ')) {
                dataLine = line.substring(6);
              }
            }
            
            if (eventType === 'metrics' && dataLine) {
              try {
                const data = JSON.parse(dataLine);
                
                if (data.vram_total_bytes !== undefined || data.vram_used_bytes !== undefined) {
                  metricsReceived = true;
                  log('Metrics Stream 데이터 수신 성공', 'success');
                  log(`Metrics Stream 데이터: ${JSON.stringify(data, null, 2)}`, 'info');
                  
                  clearTimeout(timeout);
                  req.destroy();
                  resolve(true);
                  return;
                }
              } catch (error) {
                // JSON 파싱 오류 무시
              }
            }
          }
        });

        res.on('end', () => {
          clearTimeout(timeout);
          if (!metricsReceived) {
            reject(new Error('Metrics Stream에서 데이터를 받지 못했습니다'));
          }
        });

        res.on('error', (error) => {
          clearTimeout(timeout);
          logError('Metrics Stream 응답 오류', error);
          reject(error);
        });

        // 타임아웃 설정 (5초)
        timeout = setTimeout(() => {
          req.destroy();
          if (!metricsReceived) {
            reject(new Error('Metrics Stream 타임아웃'));
          }
        }, 5000);
      });

      req.on('error', (error) => {
        logError('Metrics Stream 요청 실패', error);
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('Metrics Stream 타임아웃'));
      });

      req.end();

    } catch (error) {
      logError('Metrics Stream 테스트 실패', error);
      reject(error);
    }
  });
}

// 추론 테스트
async function testInference() {
  log(`추론 테스트 시작: "${TEST_PROMPT}"`);
  
  return new Promise((resolve, reject) => {
    try {
      const postData = JSON.stringify({
        prompt: TEST_PROMPT,
        stream: true,
        temperature: 0.7,
        top_k: 40,
        top_p: 0.95,
        min_p: 0.05,
        max_tokens: 100
      });

      const options = {
        hostname: 'localhost',
        port: SERVER_PORT,
        path: '/completion',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(postData)
        },
        timeout: INFERENCE_TIMEOUT
      };

      const req = http.request(options, (res) => {
        let responseData = '';
        let tokensReceived = 0;
        let hasError = false;

        res.on('data', (chunk) => {
          responseData += chunk.toString();
          
          // SSE 형식 파싱
          const lines = responseData.split('\n');
          responseData = lines.pop() || ''; // 마지막 불완전한 라인 보관
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                
                if (data.error) {
                  hasError = true;
                  logError(`추론 오류: ${data.error}`);
                } else if (data.content) {
                  tokensReceived++;
                  process.stdout.write(data.content);
                } else if (data.stop) {
                  process.stdout.write('\n');
                  if (!hasError && tokensReceived > 0) {
                    log(`추론 테스트 성공 (토큰 수: ${tokensReceived})`, 'success');
                    resolve(true);
                  } else {
                    reject(new Error('추론 실패: 토큰을 받지 못했습니다'));
                  }
                }
              } catch (e) {
                // JSON 파싱 오류 무시 (불완전한 데이터일 수 있음)
              }
            }
          }
        });

        res.on('end', () => {
          if (!hasError && tokensReceived > 0) {
            log(`추론 테스트 성공 (토큰 수: ${tokensReceived})`, 'success');
            resolve(true);
          } else if (!hasError) {
            reject(new Error('추론 실패: 응답을 받지 못했습니다'));
          }
        });
      });

      req.on('error', (error) => {
        logError('추론 요청 실패', error);
        reject(error);
      });

      req.on('timeout', () => {
        req.destroy();
        reject(new Error('추론 타임아웃'));
      });

      req.write(postData);
      req.end();

    } catch (error) {
      logError('추론 테스트 실패', error);
      reject(error);
    }
  });
}

// C++ 파일 변경 감지
function watchCppFile() {
  const cppFilePath = path.join(__dirname, 'src', 'mlx_server.cpp');
  
  if (!fs.existsSync(cppFilePath)) {
    log('C++ 파일을 찾을 수 없습니다', 'warn');
    return;
  }

  try {
    const stats = fs.statSync(cppFilePath);
    lastCppFileMtime = stats.mtimeMs;
  } catch (error) {
    logError('C++ 파일 상태 확인 실패', error);
    return;
  }

  cppFileWatcher = fs.watchFile(cppFilePath, { interval: 1000 }, async (curr, prev) => {
    if (curr.mtimeMs !== prev.mtimeMs && curr.mtimeMs !== lastCppFileMtime) {
      lastCppFileMtime = curr.mtimeMs;
      log('C++ 파일 변경 감지됨', 'warn');
      await rebuildAndRestart();
    }
  });

  log('C++ 파일 감시 시작됨');
}

// 재컴파일 및 서버 재시작
async function rebuildAndRestart() {
  log('C++ 모듈 재컴파일 시작...');
  
  return new Promise((resolve, reject) => {
    const buildProcess = spawn('npm', ['run', 'build'], {
      cwd: __dirname,
      stdio: 'inherit',
      shell: true
    });

    buildProcess.on('exit', async (code) => {
      if (code === 0) {
        log('컴파일 성공', 'success');
        log('서버 재시작 중...');
        try {
          await startServer();
          log('서버 재시작 완료', 'success');
          resolve();
        } catch (error) {
          logError('서버 재시작 실패', error);
          reject(error);
        }
      } else {
        logError(`컴파일 실패 (코드: ${code})`);
        reject(new Error(`컴파일 실패: ${code}`));
      }
    });

    buildProcess.on('error', (error) => {
      logError('컴파일 프로세스 오류', error);
      reject(error);
    });
  });
}

// 전체 테스트 실행 (1번만 실행)
async function runTest() {
  testCount++;
  log(`\n========== 테스트 #${testCount} 시작 ==========`);
  
  try {
    // Health check
    await testHealthCheck();
    
    // Metrics
    await testMetrics();
    
    // Metrics Stream
    await testMetricsStream();
    
    // Inference
    await testInference();
    
    successCount++;
    log(`\n========== 테스트 #${testCount} 성공 ==========`, 'success');
    log(`성공: ${successCount}, 실패: ${failureCount}`);
    log('테스트 완료 - 1번만 실행됨');
    
    // 테스트 완료 후 종료
    await cleanup();
    process.exit(0);
    
  } catch (error) {
    failureCount++;
    logError(`\n========== 테스트 #${testCount} 실패 ==========`, error);
    log(`성공: ${successCount}, 실패: ${failureCount}`);
    log('테스트 실패 - 재시도하지 않음');
    
    // 실패 시에도 재시도하지 않고 종료
    await cleanup();
    process.exit(1);
  }
}

// 정리 함수
async function cleanup() {
  log('정리 중...');
  
  if (cppFileWatcher) {
    fs.unwatchFile(path.join(__dirname, 'src', 'mlx_server.cpp'));
    cppFileWatcher = null;
  }
  
  await stopServer();
  
  // 임시 파일 삭제
  const tempScriptPath = path.join(__dirname, 'test-server-temp.js');
  if (fs.existsSync(tempScriptPath)) {
    fs.unlinkSync(tempScriptPath);
  }
}

// 프로세스 종료 핸들러
process.on('SIGINT', async () => {
  log('\n프로세스 종료 신호 수신');
  await cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  log('\n프로세스 종료 신호 수신');
  await cleanup();
  process.exit(0);
});

// 메인 실행
async function main() {
  log('MLX 서버 테스트 시작 (1번만 실행)');
  log(`테스트 질의어: "${TEST_PROMPT}"`);
  log(`서버 URL: ${SERVER_URL}`);
  
  try {
    // C++ 파일 감시 시작
    watchCppFile();
    
    // 서버 시작
    await startServer();
    
    // 테스트 시작
    await runTest();
    
  } catch (error) {
    logError('메인 실행 오류', error);
    await cleanup();
    process.exit(1);
  }
}

// fetch는 사용하지 않음 (http 모듈 사용)

// 실행
main().catch(async (error) => {
  logError('치명적 오류', error);
  await cleanup();
  process.exit(1);
});

