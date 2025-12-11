const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const os = require('os');

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
    
    // 대략적인 추정:
    // - 모델의 일부가 GPU로 오프로드됨 (gpuLayers에 비례)
    // - KV 캐시도 VRAM 사용 (contextSize에 비례)
    // - 실제로는 모델 크기의 30-50% 정도가 VRAM에 로드됨 (양자화에 따라 다름)
    
    // 간단한 추정: 모델 크기의 일부 + KV 캐시
    // 실제로는 양자화 레벨, 모델 아키텍처 등에 따라 다르지만, 대략적인 추정
    const estimatedModelVRAM = modelSizeBytes * 0.4; // 모델의 40% 정도가 VRAM에 로드된다고 가정
    
    // KV 캐시 추정 (매우 간단한 추정)
    const contextSize = currentModelConfig.contextSize || 2048;
    // 대략적으로 context당 1KB 정도 (실제로는 모델 크기에 비례)
    const estimatedKVCache = contextSize * 1024 * 2; // 2배 여유
    
    const totalEstimated = estimatedModelVRAM + estimatedKVCache;
    
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
  
  const args = ['-m', modelPath];
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
    
    // VRAM 사용량 파싱 (llama-server 로그에서 추출)
    // 예: "llm_load_tensors: using Metal backend" 또는 "ggml_metal: allocated buffer" 등
    // Metal 백엔드 사용 시 VRAM 정보가 로그에 포함될 수 있음
    // 더 정확한 방법: llama-server의 /metrics 엔드포인트 사용 (구현 필요)
    parseVRAMUsage(msg);
  });
  llamaServerProcess.stderr.on('data', (data) => {
    const msg = data.toString();
    console.error(msg);
    sendLog('log-message', `[STDERR] ${msg}`);
    
    // stderr에서도 VRAM 정보 파싱 시도
    parseVRAMUsage(msg);
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
      if (cachedVramTotal > 0) {
        // VRAM 사용량 추정
        const estimatedVRAMUsed = estimateVRAMUsage();
        if (estimatedVRAMUsed > 0) {
          gpuUsage = (estimatedVRAMUsed / cachedVramTotal) * 100;
          cachedVramUsed = estimatedVRAMUsed; // 캐시 업데이트
        } else {
          // 모델이 로드되지 않았거나 GPU 레이어가 0이면 0%
          gpuUsage = 0;
        }
      } else {
        // VRAM 총량을 알 수 없으면 랜덤값 사용 (임시)
        gpuUsage = Math.random() * 100;
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
