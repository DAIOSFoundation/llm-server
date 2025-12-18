#!/usr/bin/env node

const http = require('http');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

// EventSource는 사용하지 않음 (HTTP 스트림 직접 파싱)

// 설정
const SERVER_PORT = 8081;
const SERVER_URL = `http://localhost:${SERVER_PORT}`;
const TEST_PROMPT = '안녕, 대한민국의 수도는 어디지?';
const MAX_RETRIES = 100; // 최대 재시도 횟수 (무한 반복을 위해 큰 값)
const HEALTH_CHECK_RETRIES = 120; // 모델 로딩을 위해 더 많은 재시도 (최대 4분)
const HEALTH_CHECK_DELAY = 2000; // 2초
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

      // Python 직접 HTTP 서버 사용 (FastAPI 기반)
      const serverScriptPath = path.join(__dirname, 'server-python-direct.py');
      
      if (!fs.existsSync(serverScriptPath)) {
        throw new Error('server-python-direct.py를 찾을 수 없습니다');
      }

      // 모델 경로를 환경변수로 설정
      const modelPath = path.isAbsolute(mlxModel.modelPath) 
        ? mlxModel.modelPath 
        : path.join(__dirname, 'models', mlxModel.modelPath);
      
      // Python 직접 HTTP 서버 시작 (venv 사용)
      log('Python 직접 HTTP 서버 시작 중...');
      const venvPython = path.join(__dirname, 'venv', 'bin', 'python3');
      const pythonCmd = fs.existsSync(venvPython) ? venvPython : 'python3';
      
      serverProcess = spawn(pythonCmd, [serverScriptPath], {
        cwd: __dirname,
        stdio: 'pipe',
        env: { 
          ...process.env,
          MLX_MODEL_PATH: modelPath,
          PORT: SERVER_PORT.toString()
        }
      });

      let serverOutput = '';
      let serverOutputLines = [];
      serverProcess.stdout.on('data', (data) => {
        const output = data.toString();
        serverOutput += output;
        const lines = output.split('\n');
        serverOutputLines.push(...lines.filter(l => l.trim()));
        process.stdout.write(`[Server] ${output}`);
      });

      serverProcess.stderr.on('data', (data) => {
        const output = data.toString();
        serverOutput += output;
        const lines = output.split('\n');
        serverOutputLines.push(...lines.filter(l => l.trim()));
        process.stderr.write(`[Server Error] ${output}`);
      });

      serverProcess.on('exit', (code, signal) => {
        if (code !== 0 && code !== null) {
          logError(`서버가 종료되었습니다 (코드: ${code}, signal: ${signal})`);
          log('서버 출력:', 'info');
          console.log(serverOutput);
          // 서버가 크래시된 경우 reject (FastAPI/Uvicorn 시작 메시지 확인)
          if (!serverOutput.includes('Uvicorn running') && !serverOutput.includes('Application startup complete')) {
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
        
        // Health check로 서버 시작 확인 (출력 메시지보다 더 신뢰할 수 있음)
        try {
          const response = await new Promise((resolve, reject) => {
            const req = http.get(`${SERVER_URL}/health`, (res) => {
              let data = '';
              res.on('data', (chunk) => { data += chunk; });
              res.on('end', () => {
                resolve({ statusCode: res.statusCode, data });
              });
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
          
          // 서버가 응답하면 시작된 것으로 간주 (200, 503 등 어떤 상태 코드든)
          clearInterval(checkInterval);
          log(`서버가 시작되었습니다 (HTTP ${response.statusCode})`);
          serverStarted = true;
          resolve();
        } catch (error) {
          // 서버가 아직 시작되지 않음 - 정상적인 상황
          if (Date.now() - startTime > SERVER_START_TIMEOUT) {
            clearInterval(checkInterval);
            const recentOutput = serverOutputLines.slice(-10).join('\n');
            reject(new Error(`서버 시작 타임아웃 (${SERVER_START_TIMEOUT}ms). 최근 출력:\n${recentOutput || '(출력 없음)'}`));
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
      
      // Python 기반 서버는 'ready' 또는 'loading' 상태 반환
      if (data.status === 'ok' || data.status === 'ready') {
        log('Health check 성공 (모델 준비됨)', 'success');
        return true;
      } else if (data.status === 'loading') {
        // 모델이 아직 로딩 중이면 재시도 (최대 30회, 총 60초)
        if (i < HEALTH_CHECK_RETRIES - 1) {
          log(`모델 로딩 중... (재시도 ${i + 1}/${HEALTH_CHECK_RETRIES})`, 'warn');
          throw new Error('Model is still loading');
        } else {
          // 최대 재시도 횟수에 도달했지만 여전히 로딩 중이면 에러
          throw new Error('모델 로딩 타임아웃: 모델이 준비되지 않았습니다');
        }
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

    // Python 기반 서버는 JSON 형식의 메트릭 반환
    try {
      const data = JSON.parse(response.data);
      if (data.engine === 'python-mlx-lm' || data.ready !== undefined) {
        log('Metrics 엔드포인트 성공', 'success');
        log(`Metrics 내용: ${JSON.stringify(data, null, 2)}`, 'info');
        return true;
      } else {
        throw new Error('예상하지 못한 metrics 형식');
      }
    } catch (e) {
      // JSON 파싱 실패 시 텍스트 형식으로 시도
      const text = response.data;
      if (text.includes('vram_total_bytes') || text.includes('vram_used_bytes')) {
        log('Metrics 엔드포인트 성공', 'success');
        log(`Metrics 내용:\n${text}`, 'info');
        return true;
      } else {
        throw new Error('예상하지 못한 metrics 형식');
      }
    }
  } catch (error) {
    logError('Metrics 테스트 실패', error);
    throw error;
  }
}

// Metrics Stream 테스트 (Python 기반 서버는 지원하지 않음 - 건너뜀)
async function testMetricsStream() {
  log('Metrics Stream 테스트 건너뜀 (Python 기반 서버는 지원하지 않음)', 'warn');
  return true;
}

// 추론 테스트
async function testInference() {
  log(`추론 테스트 시작: "${TEST_PROMPT}"`);
  
  return new Promise((resolve, reject) => {
    try {
      const postData = JSON.stringify({
        prompt: TEST_PROMPT,
        temperature: 0.7,
        // top_k: 40,  // mlx_lm에서 지원하지 않으므로 제거
        top_p: 0.95,
        max_tokens: 100
      });

      const options = {
        hostname: 'localhost',
        port: SERVER_PORT,
        path: '/chat',  // Python 기반 서버는 /chat 사용
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

        if (res.statusCode !== 200) {
          let errorBody = '';
          res.on('data', (chunk) => { errorBody += chunk.toString(); });
          res.on('end', () => {
            logError(`HTTP ${res.statusCode}: ${errorBody}`);
            reject(new Error(`HTTP ${res.statusCode}`));
          });
          return;
        }

        res.on('data', (chunk) => {
          const chunkStr = chunk.toString();
          responseData += chunkStr;
          
          // Python 기반 서버는 일반 텍스트 스트리밍 (SSE 아님)
          // 각 chunk를 직접 출력
          process.stdout.write(chunkStr);
          tokensReceived += chunkStr.length;
        });

        res.on('end', () => {
          process.stdout.write('\n');
          if (!hasError && tokensReceived > 0) {
            log(`추론 테스트 성공 (응답 길이: ${tokensReceived} bytes)`, 'success');
            resolve(true);
          } else if (!hasError) {
            reject(new Error('추론 실패: 응답을 받지 못했습니다'));
          }
        });

        res.on('error', (error) => {
          logError('응답 스트림 오류', error);
          reject(error);
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

// 전체 테스트 실행 (1번만 수행)
async function runTest() {
  testCount++;
  log(`\n========== 테스트 #${testCount} 시작 ==========`);
  
  try {
    // Health check
    await testHealthCheck();
    
    // Metrics
    await testMetrics();
    
    // Metrics Stream (Python 기반 서버는 지원하지 않음 - 건너뜀)
    await testMetricsStream();
    
    // Inference
    await testInference();
    
    successCount++;
    log(`\n========== 테스트 #${testCount} 성공 ==========`, 'success');
    log(`성공: ${successCount}, 실패: ${failureCount}`);
    
    // 테스트 완료 (1번만 수행)
    log('테스트 완료', 'success');
    
  } catch (error) {
    failureCount++;
    logError(`\n========== 테스트 #${testCount} 실패 ==========`, error);
    log(`성공: ${successCount}, 실패: ${failureCount}`);
    
    // 실패 시에도 1번만 수행하고 종료
    log('테스트 실패로 종료', 'error');
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
  
  // 임시 파일 삭제는 필요 없음 (Python 기반 서버는 직접 실행)
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
  log('MLX Python 기반 서버 추론 테스트 시작 (1회 실행)');
  log(`테스트 질의어: "${TEST_PROMPT}"`);
  log(`서버 URL: ${SERVER_URL}`);
  
  try {
    // C++ 파일 감시는 비활성화 (1회 실행이므로)
    // watchCppFile();
    
    // 서버 시작
    await startServer();
    
    // 테스트 시작 (1번만 수행)
    await runTest();
    
    // 테스트 완료 후 정리
    log('모든 테스트 완료', 'success');
    await cleanup();
    process.exit(0);
    
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

