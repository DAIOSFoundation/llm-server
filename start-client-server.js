const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const url = require('url');

// 설정 파일 경로
// 클라이언트 모드에서는 프로젝트 루트의 config.json 사용
const CONFIG_PATH = path.join(__dirname, 'config.json');
const MODELS_CONFIG_PATH = path.join(__dirname, 'models-config.json');

let llamaServerProcess = null;
let mlxServerInstance = null;
let ggufModelConfig = null; // 현재 GGUF 서버에 로드된 모델
let mlxModelConfig = null; // 현재 MLX 서버에 로드된 모델

// 설정 로드
function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const data = fs.readFileSync(CONFIG_PATH, 'utf-8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('[Client Server] Failed to load config:', error);
  }
  return { models: [], activeModelId: null };
}

// GGUF 서버 종료
function stopGgufServer() {
  return new Promise((resolve) => {
    if (llamaServerProcess) {
      console.log(`[Client Server] Stopping GGUF server`);
      try {
          const processToKill = llamaServerProcess;
          console.log(`[Client Server] Attempting to kill GGUF server process`);
          console.log(`[Client Server]   Process type: ${typeof processToKill}`);
          console.log(`[Client Server]   Has kill method: ${typeof processToKill?.kill === 'function'}`);
          console.log(`[Client Server]   Process PID: ${processToKill?.pid || 'unknown'}`);
          
          if (processToKill && typeof processToKill.kill === 'function') {
            let resolved = false;
            const resolveOnce = () => {
              if (!resolved) {
                resolved = true;
                console.log(`[Client Server] ✅ llama.cpp server process terminated`);
                llamaServerProcess = null;
                ggufModelConfig = null;
                resolve();
              }
            };
            
            processToKill.once('close', (code) => {
              console.log(`[Client Server] Process close event, code: ${code}`);
              resolveOnce();
            });
            
            processToKill.once('exit', (code) => {
              console.log(`[Client Server] Process exit event, code: ${code}`);
              resolveOnce();
            });
            
            console.log(`[Client Server] Sending SIGTERM to process ${processToKill.pid}`);
            processToKill.kill('SIGTERM');
            
            setTimeout(() => {
              if (processToKill && !processToKill.killed && !resolved) {
                console.log(`[Client Server] ⚠️  Process not terminated, sending SIGKILL`);
                try {
                  processToKill.kill('SIGKILL');
                } catch (err) {
                  console.error(`[Client Server] Error killing process:`, err);
                }
              }
              setTimeout(() => {
                if (!resolved) {
                  console.log(`[Client Server] ⚠️  Force resolving after timeout`);
                  resolveOnce();
                }
              }, 500);
            }, 2000);
          } else {
            console.log(`[Client Server] ⚠️  Process object invalid, clearing state`);
            setTimeout(() => {
              llamaServerProcess = null;
              ggufModelConfig = null;
              resolve();
            }, 500);
          }
      } catch (error) {
        console.error(`[Client Server] Error stopping GGUF server:`, error);
        llamaServerProcess = null;
        ggufModelConfig = null;
        resolve();
      }
    } else {
      resolve();
    }
  });
}

// MLX 서버 종료
function stopMlxServer() {
  return new Promise((resolve) => {
    if (mlxServerInstance) {
      console.log(`[Client Server] Stopping MLX server`);
      mlxServerInstance.stop().then(() => {
        mlxServerInstance = null;
        mlxModelConfig = null;
        resolve();
      }).catch((error) => {
        console.error(`[Client Server] Error stopping MLX server:`, error);
        mlxServerInstance = null;
        mlxModelConfig = null;
        resolve();
      });
    } else {
      resolve();
    }
  });
}

// GGUF 서버 시작
function startGgufServer(modelConfig) {
  console.log(`[Client Server] ===== GGUF SERVER START =====`);
  const { modelPath, id, contextSize, gpuLayers } = modelConfig;
  
  console.log(`[Client Server] Model ID: ${id}`);
  console.log(`[Client Server] Model Path (raw): ${modelPath}`);
  
  // modelPath가 상대 경로인 경우 절대 경로로 변환
  let absoluteModelPath = modelPath;
  if (modelPath && !path.isAbsolute(modelPath)) {
    // models-config.json에서 가져온 경우 llama.cpp/models/ 기준
    let modelFileName = modelPath;
    // .gguf 확장자가 없으면 추가
    if (!modelFileName.endsWith('.gguf')) {
      modelFileName = modelFileName + '.gguf';
      console.log(`[Client Server] Added .gguf extension: ${modelFileName}`);
    }
    absoluteModelPath = path.resolve(__dirname, 'llama.cpp', 'models', modelFileName);
    console.log(`[Client Server] Converted to absolute path: ${absoluteModelPath}`);
  }
  
  if (!absoluteModelPath || !fs.existsSync(absoluteModelPath)) {
    console.error(`[Client Server] ❌ Model path "${absoluteModelPath}" is invalid or not found`);
    console.error(`[Client Server]   Original path: ${modelPath}`);
    console.error(`[Client Server]   Absolute path: ${absoluteModelPath}`);
    // .gguf 확장자를 제거하고 다시 시도
    if (absoluteModelPath.endsWith('.gguf')) {
      const withoutExt = absoluteModelPath.slice(0, -5);
      console.error(`[Client Server]   Trying without extension: ${withoutExt}`);
      if (fs.existsSync(withoutExt)) {
        absoluteModelPath = withoutExt;
        console.log(`[Client Server] ✅ Found model without .gguf extension`);
      } else {
        return;
      }
    } else {
      return;
    }
  }
  
  console.log(`[Client Server] ✅ Model file found: ${absoluteModelPath}`);

  const serverExecutable = path.resolve(__dirname, 'llama.cpp', 'build', 'bin', 'llama-server');
  
  if (!fs.existsSync(serverExecutable)) {
    console.error(`[Client Server] ❌ llama-server executable not found at: ${serverExecutable}`);
    return;
  }
  
  console.log(`[Client Server] ✅ Server executable found: ${serverExecutable}`);
  
  const args = ['-m', absoluteModelPath, '--metrics', '--port', '8080'];
  if (contextSize) args.push('-c', contextSize.toString());
  if (gpuLayers !== undefined && gpuLayers !== null && gpuLayers >= 0) {
    args.push('-ngl', gpuLayers.toString());
  }

  console.log(`[Client Server] 🚀 Spawning process: ${serverExecutable}`);
  console.log(`[Client Server]    Args: ${args.join(' ')}`);
  
  llamaServerProcess = spawn(serverExecutable, args);
  
  // 프로세스가 즉시 종료되는 경우 감지
  let processStarted = false;
  const startTimeout = setTimeout(() => {
    if (!processStarted && llamaServerProcess && llamaServerProcess.killed) {
      console.error(`[Client Server] llama-server process failed to start`);
    }
  }, 3000);

  llamaServerProcess.stdout.on('data', (data) => {
    processStarted = true;
    clearTimeout(startTimeout);
    const output = data.toString();
    console.log(`[GGUF Server] ${output}`);
    // 서버가 시작되었는지 확인
    if (output.includes('listening') || output.includes('port') || output.includes('HTTP server listening')) {
      console.log(`[Client Server] ✅ GGUF server started successfully and listening on port 8080`);
      console.log(`[Client Server]    Model: ${id}`);
      console.log(`[Client Server]    Path: ${absoluteModelPath}`);
    }
  });
  
  llamaServerProcess.stderr.on('data', (data) => {
    processStarted = true;
    clearTimeout(startTimeout);
    const output = data.toString();
    console.error(`[GGUF Server] ${output}`);
  });
  
  llamaServerProcess.on('close', (code) => {
    clearTimeout(startTimeout);
    console.log(`[Client Server] ⚠️  llama-server process exited with code ${code}`);
    llamaServerProcess = null;
    ggufModelConfig = null;
    console.log(`[Client Server]    Server state cleared`);
  });
  
  llamaServerProcess.on('error', (error) => {
    clearTimeout(startTimeout);
    console.error(`[Client Server] ❌ Failed to spawn llama-server:`, error);
    llamaServerProcess = null;
    ggufModelConfig = null;
  });

  ggufModelConfig = modelConfig;
  console.log(`[Client Server] ✅ Server state updated: type=gguf, model=${id}`);
  console.log(`[Client Server]    Process PID: ${llamaServerProcess.pid || 'unknown'}`);
  console.log(`[Client Server] ===== GGUF SERVER START COMPLETE =====`);
}

// MLX 서버 시작
async function startMlxServer(modelConfig) {
  console.log(`[Client Server] ===== MLX SERVER START =====`);
  console.log(`[Client Server] Model ID: ${modelConfig.id}`);
  console.log(`[Client Server] Model Path: ${modelConfig.modelPath}`);
  
  const MlxServer = require(path.join(__dirname, 'mlx', 'server'));
  
  try {
    currentModelConfig = modelConfig;
    console.log(`[Client Server] Creating MLX server instance...`);
    
    const mlxServer = new MlxServer(modelConfig);
    console.log(`[Client Server] Starting MLX server (async)...`);
    await mlxServer.start();
    
    console.log(`[Client Server] ✅ MLX server started successfully`);
    mlxServerInstance = mlxServer;
    
    mlxModelConfig = modelConfig;
    console.log(`[Client Server] ✅ Server state updated: type=mlx, model=${modelConfig.id}`);
    console.log(`[Client Server]    Model path: ${modelConfig.modelPath}`);
    console.log(`[Client Server] ===== MLX SERVER START COMPLETE =====`);
  } catch (error) {
    console.error(`[Client Server] ❌ Failed to start MLX server:`, error.message);
    console.error(`[Client Server]    Error details:`, error);
    mlxModelConfig = null;
    mlxServerInstance = null;
  }
}

// 서버 시작 (형식에 따라)
function startServerByFormat(modelConfig) {
  const { modelFormat, id } = modelConfig;
  const format = modelFormat || 'gguf';
  console.log(`[Client Server] 📋 Starting server by format: ${format}`);
  console.log(`[Client Server]   Model: ${id}`);

  if (format === 'mlx') {
    console.log(`[Client Server] 🍎 Starting MLX server...`);
    startMlxServer(modelConfig);
  } else {
    console.log(`[Client Server] 🦙 Starting GGUF (llama.cpp) server...`);
    startGgufServer(modelConfig);
  }
}

// 모든 서버 시작 (초기 로드 시 GGUF와 MLX 서버를 모두 시작)
async function startAllServers(config) {
  console.log(`[Client Server] ==========================================`);
  console.log(`[Client Server] ===== STARTING ALL SERVERS =====`);
  console.log(`[Client Server] ==========================================`);
  
  if (!config || !config.models || config.models.length === 0) {
    console.log('[Client Server] ⚠️  No models in config, skipping server start');
    return;
  }

  // GGUF 모델 찾기
  const ggufModel = config.models.find(m => (m.modelFormat || 'gguf') === 'gguf');
  // MLX 모델 찾기
  const mlxModel = config.models.find(m => m.modelFormat === 'mlx');

  // GGUF 서버 시작
  if (ggufModel) {
    if (!llamaServerProcess) {
      console.log(`[Client Server] 🚀 Starting GGUF server for model: ${ggufModel.id}`);
      startGgufServer(ggufModel);
    } else {
      console.log(`[Client Server] ⏭️  GGUF server already running`);
    }
  } else {
    console.log(`[Client Server] ⚠️  No GGUF model found in config`);
  }

  // MLX 서버 시작
  if (mlxModel) {
    if (!mlxServerInstance) {
      console.log(`[Client Server] 🚀 Starting MLX server for model: ${mlxModel.id}`);
      await startMlxServer(mlxModel);
    } else {
      console.log(`[Client Server] ⏭️  MLX server already running`);
    }
  } else {
    console.log(`[Client Server] ⚠️  No MLX model found in config`);
  }

  console.log(`[Client Server] ===== ALL SERVERS START COMPLETE =====`);
}

// 설정 파일 감시 및 서버 시작 (초기 로드 시에만 서버 시작)
let isInitialLoad = true;
function watchConfigAndStartServer() {
  console.log('[Client Server] ===== Config Watch Triggered =====');
  console.log('[Client Server][DEBUG] watchConfigAndStartServer called at:', new Date().toISOString());
  const config = loadConfig();
  console.log('[Client Server] Config loaded:');
  console.log('[Client Server]    Active Model ID:', config.activeModelId);
  console.log('[Client Server]    Models count:', config.models?.length || 0);
  console.log('[Client Server][DEBUG] Config object:', JSON.stringify(config, null, 2));
  
  // 초기 로드 시에만 모든 서버 시작
  if (isInitialLoad && config.models && config.models.length > 0) {
    console.log('[Client Server] 🚀 Initial load: Starting all servers...');
    startAllServers(config);
    isInitialLoad = false;
  } else {
    console.log('[Client Server] ⏭️  Config changed, but skipping server restart (servers already running)');
  }
  
  console.log('[Client Server][DEBUG] watchConfigAndStartServer completed');
}

// 설정 파일 변경 감시
let configWatcher = null;
function startWatchingConfig() {
  if (fs.existsSync(CONFIG_PATH)) {
    if (configWatcher) {
      fs.unwatchFile(CONFIG_PATH);
    }
    configWatcher = fs.watchFile(CONFIG_PATH, { interval: 1000 }, (curr, prev) => {
      if (curr.mtime !== prev.mtime) {
        console.log('[Client Server] Config file changed, reloading...');
        watchConfigAndStartServer();
      }
    });
  } else {
    // 파일이 없으면 주기적으로 확인
    setTimeout(() => {
      if (fs.existsSync(CONFIG_PATH)) {
        startWatchingConfig();
        watchConfigAndStartServer();
      } else {
        startWatchingConfig();
      }
    }, 2000);
  }
}

// 초기 서버 시작
watchConfigAndStartServer();

// 설정 파일 감시 시작
startWatchingConfig();

  // 프로세스 종료 처리
process.on('SIGTERM', () => {
  console.log('[Client Server] Shutting down...');
  if (configWatcher) {
    fs.unwatchFile(CONFIG_PATH);
  }
  Promise.all([stopGgufServer(), stopMlxServer()]).then(() => {
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('[Client Server] Shutting down...');
  if (configWatcher) {
    fs.unwatchFile(CONFIG_PATH);
  }
  Promise.all([stopGgufServer(), stopMlxServer()]).then(() => {
    process.exit(0);
  });
});

// HTTP 서버 시작 (설정 저장용)
const httpServer = http.createServer((req, res) => {
  // CORS 헤더 설정
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);
  
  // /api/save-config - 설정 저장
  if (parsedUrl.pathname === '/api/save-config' && req.method === 'POST') {
    console.log('[Client Server] ===== API: /api/save-config REQUEST =====');
    console.log('[Client Server][DEBUG] Request received at:', new Date().toISOString());
    let body = '';
    req.on('data', chunk => { 
      body += chunk.toString();
      console.log('[Client Server][DEBUG] Received chunk, body length:', body.length);
    });
    req.on('end', () => {
      console.log('[Client Server][DEBUG] Request body received, total length:', body.length);
      try {
        const config = JSON.parse(body);
        console.log('[Client Server] 📝 Config received:');
        console.log('[Client Server]    Active Model ID:', config.activeModelId);
        console.log('[Client Server]    Models count:', config.models?.length || 0);
        const activeModel = config.models?.find(m => m.id === config.activeModelId);
        if (activeModel) {
          console.log('[Client Server]    Active Model Format:', activeModel.modelFormat || 'gguf');
          console.log('[Client Server]    Active Model Path:', activeModel.modelPath);
          console.log('[Client Server]    Active Model ID:', activeModel.id);
        } else {
          console.error('[Client Server][DEBUG] ❌ Active model not found in models array');
          console.error('[Client Server][DEBUG]    Looking for ID:', config.activeModelId);
          console.error('[Client Server][DEBUG]    Available IDs:', config.models?.map(m => m.id) || []);
        }
        
        console.log('[Client Server][DEBUG] Writing config to file:', CONFIG_PATH);
        fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), 'utf-8');
        console.log('[Client Server] ✅ Config saved to file:', CONFIG_PATH);
        console.log('[Client Server][DEBUG] Config file written, calling watchConfigAndStartServer...');
        console.log('[Client Server] 🔄 Triggering server restart...');
        
        // config만 저장하고 서버는 재시작하지 않음 (이미 실행 중인 서버 사용)
        console.log('[Client Server][DEBUG] Config saved, servers should already be running');
        console.log('[Client Server][DEBUG]   Active model ID:', config.activeModelId);
        console.log('[Client Server][DEBUG]   GGUF server running:', !!llamaServerProcess);
        console.log('[Client Server][DEBUG]   MLX server running:', !!mlxServerInstance);
        
        console.log('[Client Server][DEBUG] Server start requested, sending response...');
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: true }));
        console.log('[Client Server] ✅ API response sent: success');
      } catch (error) {
        console.error('[Client Server][DEBUG] ❌ Error in /api/save-config handler:', error);
        console.error('[Client Server] ❌ Failed to save config:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: error.message }));
      }
    });
    req.on('error', (error) => {
      console.error('[Client Server][DEBUG] ❌ Request error:', error);
    });
    return;
  }

  // /api/start-server - 서버 시작 요청 (서버가 없을 때)
  if (parsedUrl.pathname === '/api/start-server' && req.method === 'POST') {
    console.log('[Client Server] ===== API: /api/start-server REQUEST =====');
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const config = body ? JSON.parse(body) : loadConfig();
        console.log('[Client Server] 📝 Start server request received');
        console.log('[Client Server]    Active Model ID:', config.activeModelId);
        
        if (config.activeModelId) {
          const activeModel = config.models?.find(m => m.id === config.activeModelId);
          if (activeModel) {
            console.log('[Client Server] 🚀 Starting server for active model...');
            watchConfigAndStartServer();
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, message: 'Server start requested' }));
          } else {
            console.error('[Client Server] ❌ Active model not found');
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: 'Active model not found' }));
          }
        } else {
          console.error('[Client Server] ❌ No active model ID');
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: 'No active model ID' }));
        }
      } catch (error) {
        console.error('[Client Server] ❌ Failed to start server:', error);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: error.message }));
      }
    });
    return;
  }

  // 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not_found' }));
});

const HTTP_PORT = 8083; // 클라이언트 서버 관리자는 8083 포트 사용
httpServer.listen(HTTP_PORT, () => {
  console.log(`[Client Server] HTTP API server started on port ${HTTP_PORT}`);
});

console.log('[Client Server] Started. Watching config file for changes...');
console.log(`[Client Server] Config file path: ${CONFIG_PATH}`);

