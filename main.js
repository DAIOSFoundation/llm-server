const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const os = require('os');
const http = require('http');
const { promisify } = require('util');

// Metal VRAM 모니터 모듈 로드 (macOS에서만)
let metalVRAM = null;
if (process.platform === 'darwin') {
  try {
    metalVRAM = require('./native');
  } catch (error) {
    console.warn('[Main] Failed to load Metal VRAM module:', error.message);
    console.warn('[Main] Falling back to estimated VRAM usage');
  }
}

const configPath = path.join(app.getPath('userData'), 'config.json');
let llamaServerProcess = null;
let mainWindow = null;
let cachedVramTotal = 0; // VRAM 용량 캐싱
let cachedVramUsed = 0; // VRAM 사용량 (바이트)
let currentModelConfig = null; // 현재 로드된 모델 설정

// macOS VRAM 정보 가져오기
function initVRAMInfo() {
  if (process.platform === 'darwin') {
    exec('system_profiler SPDisplaysDataType', (error, stdout, stderr) => {
      if (!error) {
        // 예: "VRAM (Total): 18 GB" 또는 "VRAM (Total): 1536 MB"
        const matchGB = stdout.match(/VRAM \(Total\): (\d+) GB/);
        const matchMB = stdout.match(/VRAM \(Total\): (\d+) MB/);
        
        if (matchGB && matchGB[1]) {
          cachedVramTotal = parseInt(matchGB[1]) * 1024 * 1024 * 1024;
        } else if (matchMB && matchMB[1]) {
          cachedVramTotal = parseInt(matchMB[1]) * 1024 * 1024;
        }
        console.log(`[Main] Detected VRAM: ${(cachedVramTotal / 1024 / 1024).toFixed(0)} MB`);
      }
    });
  }
}

// VRAM 사용량 추정 (모델 크기와 GPU 레이어 수 기반)
function estimateVRAMUsage() {
  if (!currentModelConfig || !currentModelConfig.modelPath || cachedVramTotal === 0) {
    return 0;
  }

  try {
    // 모델 파일 크기 확인
    const stats = fs.statSync(currentModelConfig.modelPath);
    const modelSizeBytes = stats.size;
    
    // GPU 레이어 수 (0이면 CPU만 사용, VRAM 사용 없음)
    const gpuLayers = currentModelConfig.gpuLayers || 0;
    if (gpuLayers === 0) {
      return 0; // CPU만 사용하면 VRAM 사용 없음
    }
    
    // 더 정확한 VRAM 사용량 추정:
    // GPU 레이어 수에 따라 모델의 일부가 VRAM에 로드됨
    // 일반적으로 전체 레이어 수는 모델 크기로 추정:
    // - 7B 모델: 약 32-40 레이어
    // - 13B 모델: 약 40-48 레이어  
    // - 70B 모델: 약 80 레이어
    
    // 모델 크기로 대략적인 전체 레이어 수 추정
    const modelSizeGB = modelSizeBytes / (1024 * 1024 * 1024);
    let estimatedTotalLayers = 32; // 기본값 (7B 모델 기준)
    
    if (modelSizeGB >= 60) {
      estimatedTotalLayers = 80; // 70B 모델
    } else if (modelSizeGB >= 20) {
      estimatedTotalLayers = 48; // 13B 모델
    } else if (modelSizeGB >= 10) {
      estimatedTotalLayers = 40; // 7B 모델 (큰 버전)
    } else if (modelSizeGB >= 4) {
      estimatedTotalLayers = 32; // 7B 모델
    } else {
      estimatedTotalLayers = 24; // 작은 모델
    }
    
    // GPU 레이어 비율에 따라 모델의 일부가 VRAM에 로드됨
    // 레이어는 모델의 대부분을 차지하므로, GPU 레이어 비율이 높을수록 더 많은 모델 부분이 로드됨
    const gpuLayerRatio = Math.min(gpuLayers / estimatedTotalLayers, 1.0);
    
    // 모델 크기 기반 추정 (양자화된 모델 기준)
    // 레이어 외에도 임베딩, 출력 레이어 등이 있으므로 최소 30%는 항상 로드됨
    // GPU 레이어 비율에 따라 추가로 로드됨
    const baseModelVRAM = modelSizeBytes * 0.3; // 기본 30% (임베딩, 출력 레이어 등)
    const layerModelVRAM = modelSizeBytes * 0.6 * gpuLayerRatio; // 레이어 부분 (60% * GPU 레이어 비율)
    const estimatedModelVRAM = baseModelVRAM + layerModelVRAM;
    
    // KV 캐시 추정
    // KV 캐시는 (hidden_size * num_layers * context_size * 2 * sizeof(float16)) 정도
    // 간단한 추정: 모델 크기의 일부를 기반으로 추정
    const contextSize = currentModelConfig.contextSize || 2048;
    // GPU 레이어 수에 비례하여 KV 캐시도 증가
    const kvCachePerLayer = (modelSizeBytes * 0.0005) * (contextSize / 2048); // 레이어당 KV 캐시
    const estimatedKVCache = kvCachePerLayer * gpuLayers * 2; // key + value
    
    // 추가 오버헤드 (버퍼, 중간 활성화 등)
    const overhead = modelSizeBytes * 0.15; // 모델 크기의 15% 오버헤드
    
    const totalEstimated = estimatedModelVRAM + estimatedKVCache + overhead;
    
    // 총 VRAM을 초과하지 않도록 제한
    return Math.min(totalEstimated, cachedVramTotal * 0.95); // 최대 95%까지만
  } catch (error) {
    console.error('[Main] Failed to estimate VRAM usage:', error);
    return 0;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.maximize();

  const isDev = !app.isPackaged;
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(__dirname, 'frontend', 'dist', 'index.html'));
  }
  
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function sendLog(channel, message) {
  if (mainWindow) {
    mainWindow.webContents.send(channel, message);
  }
}

function initializeConfig() {
  if (!fs.existsSync(configPath)) {
    const defaultConfig = { models: [], activeModelId: null };
    fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2), 'utf-8');
  }
}

function startLlamaServer(modelConfig) {
  if (llamaServerProcess) {
    llamaServerProcess.kill();
  }

  currentModelConfig = modelConfig; // 현재 모델 설정 저장
  const { modelPath, contextSize, gpuLayers, frequencyPenalty, presencePenalty } = modelConfig;

  if (!modelPath || !fs.existsSync(modelPath)) {
    const msg = `Model path "${modelPath}" is invalid. Server not started.`;
    console.log(msg);
    sendLog('log-message', `[ERROR] ${msg}`);
    return;
  }

  const isDev = !app.isPackaged;
  const serverExecutable = isDev
    ? path.resolve(__dirname, 'llama.cpp', 'build', 'bin', 'llama-server')
    : path.join(process.resourcesPath, 'bin', 'llama-server');

  if (!fs.existsSync(serverExecutable)) {
    const msg = `llama-server executable not found at: ${serverExecutable}`;
    console.error(msg);
    sendLog('log-message', `[ERROR] ${msg}`);
    dialog.showErrorBox('Server Error', msg);
    return;
  }
  
  const args = ['-m', modelPath, '--metrics', '--port', '8080']; // --metrics 플래그 추가, 포트 명시
  if (contextSize) args.push('-c', contextSize.toString());
  if (gpuLayers !== undefined && gpuLayers !== null) args.push('-ngl', gpuLayers.toString());
  if (frequencyPenalty) args.push('--frequency-penalty', frequencyPenalty.toString());
  if (presencePenalty) args.push('--presence-penalty', presencePenalty.toString());

  const commandString = `${path.basename(serverExecutable)} ${args.join(' ')}`;
  console.log(`Starting server: ${commandString}`);
  sendLog('log-message', `[INFO] Starting server: ${commandString}`);
  
  llamaServerProcess = spawn(serverExecutable, args);

  llamaServerProcess.stdout.on('data', (data) => {
    const msg = data.toString();
    console.log(msg);
    sendLog('log-message', msg);
  });
  llamaServerProcess.stderr.on('data', (data) => {
    const msg = data.toString();
    console.error(msg);
    sendLog('log-message', `[STDERR] ${msg}`);
  });
  llamaServerProcess.on('close', (code) => {
    const msg = `llama-server process exited with code ${code}`;
    console.log(msg);
    sendLog('log-message', `[INFO] ${msg}`);
    llamaServerProcess = null;
    currentModelConfig = null; // 모델 설정 초기화
    cachedVramUsed = 0; // VRAM 사용량 초기화
  });
}

app.whenReady().then(() => {
  initVRAMInfo();
  initializeConfig();

  ipcMain.handle('dialog:openFile', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'Models', extensions: ['gguf'] }],
    });
    return canceled ? null : filePaths[0];
  });

  ipcMain.handle('load-config', async () => {
    try {
      const data = fs.readFileSync(configPath, 'utf-8');
      const config = JSON.parse(data);
      if (config.activeModelId) {
        const activeModel = config.models.find(m => m.id === config.activeModelId);
        if (activeModel) {
          startLlamaServer(activeModel);
        }
      }
      return config;
    } catch (error) {
      console.error('Failed to load config:', error);
      return { models: [], activeModelId: null };
    }
  });

  ipcMain.handle('save-config', async (event, configData) => {
    try {
      fs.writeFileSync(configPath, JSON.stringify(configData, null, 2), 'utf-8');
      if (configData.activeModelId) {
        const activeModel = configData.models.find(m => m.id === configData.activeModelId);
        if (activeModel) {
          startLlamaServer(activeModel);
        }
      } else {
        if (llamaServerProcess) {
          llamaServerProcess.kill();
        }
      }
      return { success: true };
    } catch (error) {
      console.error('Failed to save config:', error);
      return { success: false, error: error.message };
    }
  });

      // 시스템 리소스 정보 제공
  ipcMain.handle('get-system-metrics', async () => {
    try {
      const totalMemory = os.totalmem();
      const freeMemory = os.freemem();
      const usedMemory = totalMemory - freeMemory;
      const memoryUsagePercent = (usedMemory / totalMemory) * 100;
      
      // CPU 사용량 (임시)
      const cpuUsage = Math.random() * 100; 
      
      // GPU 사용량: VRAM 점유율로 계산
      let gpuUsage = 0;
      
      // llama-server가 실행 중이면 /metrics 엔드포인트에서 VRAM 정보 가져오기
      if (llamaServerProcess) {
        try {
          // /metrics 엔드포인트에서 VRAM 정보 가져오기 (동기적으로 대기)
          const data = await new Promise((resolve, reject) => {
            const req = http.get('http://localhost:8080/metrics', { timeout: 1000 }, (res) => {
              let data = '';
              res.on('data', (chunk) => { data += chunk; });
              res.on('end', () => resolve(data));
            });
            req.on('error', reject);
            req.on('timeout', () => {
              req.destroy();
              reject(new Error('Timeout'));
            });
          });
          
          // Prometheus 형식 파싱
          const vramTotalMatch = data.match(/llamacpp:vram_total_bytes\s+(\d+)/);
          const vramUsedMatch = data.match(/llamacpp:vram_used_bytes\s+(\d+)/);
          const vramFreeMatch = data.match(/llamacpp:vram_free_bytes\s+(\d+)/);
          
          if (vramTotalMatch && vramUsedMatch) {
            cachedVramTotal = parseInt(vramTotalMatch[1], 10);
            cachedVramUsed = parseInt(vramUsedMatch[1], 10);
            if (cachedVramTotal > 0) {
              gpuUsage = (cachedVramUsed / cachedVramTotal) * 100;
            }
          } else if (vramFreeMatch && vramTotalMatch) {
            cachedVramTotal = parseInt(vramTotalMatch[1], 10);
            const vramFree = parseInt(vramFreeMatch[1], 10);
            cachedVramUsed = cachedVramTotal - vramFree;
            if (cachedVramTotal > 0) {
              gpuUsage = (cachedVramUsed / cachedVramTotal) * 100;
            }
          } else {
            // VRAM 정보를 찾을 수 없으면 추정값 사용
            if (cachedVramTotal > 0 && currentModelConfig) {
              const estimatedVRAMUsed = estimateVRAMUsage();
              if (estimatedVRAMUsed > 0) {
                gpuUsage = (estimatedVRAMUsed / cachedVramTotal) * 100;
                cachedVramUsed = estimatedVRAMUsed;
              }
            }
          }
        } catch (error) {
          console.error('[Main] Error fetching metrics:', error.message);
          // 에러 발생 시 추정값 사용
          if (cachedVramTotal > 0 && currentModelConfig) {
            const estimatedVRAMUsed = estimateVRAMUsage();
            if (estimatedVRAMUsed > 0) {
              gpuUsage = (estimatedVRAMUsed / cachedVramTotal) * 100;
              cachedVramUsed = estimatedVRAMUsed;
            }
          } else {
            // VRAM 정보를 가져올 수 없으면 0으로 설정
            gpuUsage = 0;
          }
        }
      } else {
        // llama-server가 실행 중이 아니면 Metal API로 확인 (다른 앱의 Metal 사용량)
        if (metalVRAM && process.platform === 'darwin') {
          try {
            const vramInfo = metalVRAM.getVRAMInfo();
            if (!vramInfo.error && vramInfo.total > 0) {
              cachedVramTotal = vramInfo.total;
              cachedVramUsed = vramInfo.used;
              gpuUsage = (vramInfo.used / vramInfo.total) * 100;
            }
          } catch (error) {
            console.error('[Main] Error getting VRAM info from Metal:', error);
          }
        }
        
        // 여전히 0이면 랜덤값 사용 (임시)
        if (gpuUsage === 0 && cachedVramTotal === 0) {
          gpuUsage = Math.random() * 100;
        }
      }
      
      return {
        cpu: Math.round(cpuUsage),
        gpu: Math.round(gpuUsage),
        memory: Math.round(memoryUsagePercent),
        totalMemory: totalMemory,
        usedMemory: usedMemory,
        freeMemory: freeMemory,
        vramTotal: cachedVramTotal, // VRAM 총량
        vramUsed: cachedVramUsed // VRAM 사용량 (추정값)
      };
    } catch (error) {
      console.error('Failed to get system metrics:', error);
      return { cpu: 0, gpu: 0, memory: 0 };
    }
  });

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (llamaServerProcess) {
    llamaServerProcess.kill();
  }
  if (process.platform !== 'darwin') app.quit();
});
