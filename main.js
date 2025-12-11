const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const os = require('os');
const http = require('http');
const { promisify } = require('util');

const configPath = path.join(app.getPath('userData'), 'config.json');
let llamaServerProcess = null;
let mainWindow = null;
let cachedVramTotal = 0; // VRAM 용량 캐싱
let cachedVramUsed = 0; // VRAM 사용량 (바이트) - 0으로 초기화
let currentModelConfig = null; // 현재 로드된 모델 설정
let lastVramUpdateTime = 0; // 마지막 VRAM 업데이트 시간 (디버깅용)

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
// NOTE: 이전에는 모델 크기와 GPU 레이어 수를 기반으로 VRAM 사용량을 \"추정\"하는
// estimateVRAMUsage() 함수를 사용했지만, 추정값이 실제 메트릭을 덮어쓰는 문제가 있어
// 현재는 완전히 제거했습니다. 이제 VRAM 정보는 오직 llama-server 의 /metrics 응답만 사용합니다.

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

// 메인 프로세스 로그를 렌더러로 전달 (개발자 도구에서 확인 가능)
function logToRenderer(level, ...args) {
  const message = args.map(arg => 
    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
  ).join(' ');
  
  // 터미널에도 출력
  if (level === 'error') {
    console.error(...args);
  } else if (level === 'warn') {
    console.warn(...args);
  } else {
    console.log(...args);
  }
  
  // 렌더러로도 전달 (개발자 도구에서 확인 가능)
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('main-log', { level, message, timestamp: new Date().toISOString() });
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
      
      // GPU 사용량: 실제 GPU 처리량 (토큰 처리 속도 기반 또는 GPU 활성화 여부)
      let gpuUsage = 0;
      
      // VRAM 사용량: 별도로 계산 (GPU 사용량과 분리)
      let vramUsagePercent = 0;
      
      // 항상 /metrics 엔드포인트에서 VRAM 정보 가져오기 시도
      try {
        const data = await new Promise((resolve, reject) => {
          const req = http.get('http://localhost:8080/metrics', { timeout: 2000 }, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
              if (res.statusCode === 200) {
                resolve(data);
              } else {
                reject(new Error(`HTTP ${res.statusCode}`));
              }
            });
          });
          req.on('error', reject);
          req.on('timeout', () => {
            req.destroy();
            reject(new Error('Timeout'));
          });
        });

        // 디버깅: 받은 데이터 확인
        const vramLines = data.split('\n').filter(line => line.includes('vram') && !line.startsWith('#'));
        console.log('[Main] VRAM lines from /metrics:', vramLines);

        // GPU 사용량: 토큰 처리 속도 기반 계산
        const tokensPerSecondMatch = data.match(/llamacpp:predicted_tokens_seconds\s+([\d.]+)/);
        if (tokensPerSecondMatch) {
          const tokensPerSecond = parseFloat(tokensPerSecondMatch[1]);
          gpuUsage = Math.min((tokensPerSecond / 100) * 100, 100);
        } else if (currentModelConfig && currentModelConfig.gpuLayers > 0) {
          gpuUsage = 50;
        }

        // VRAM 사용량: 별도로 계산
        const lines = data.split('\n');
        let vramTotalMatch = null;
        let vramUsedMatch = null;
        let vramFreeMatch = null;

        for (const line of lines) {
          if (!vramTotalMatch && line.includes('vram_total_bytes') && !line.startsWith('#')) {
            vramTotalMatch = line.match(/llamacpp:vram_total_bytes\s+([\d.e+\-]+)/);
          }
          if (!vramUsedMatch && line.includes('vram_used_bytes') && !line.startsWith('#')) {
            vramUsedMatch = line.match(/llamacpp:vram_used_bytes\s+([\d.e+\-]+)/);
          }
          if (!vramFreeMatch && line.includes('vram_free_bytes') && !line.startsWith('#')) {
            vramFreeMatch = line.match(/llamacpp:vram_free_bytes\s+([\d.e+\-]+)/);
          }
        }

        console.log('[Main] VRAM parsing - Total match:', vramTotalMatch ? vramTotalMatch[1] : 'null');
        console.log('[Main] VRAM parsing - Used match:', vramUsedMatch ? vramUsedMatch[1] : 'null');
        console.log('[Main] VRAM parsing - Free match:', vramFreeMatch ? vramFreeMatch[1] : 'null');

        if (vramTotalMatch && vramUsedMatch) {
          const parsedTotal = Math.round(parseFloat(vramTotalMatch[1]));
          const parsedUsed = Math.round(parseFloat(vramUsedMatch[1]));
          console.log('[Main] VRAM parsed - Total:', parsedTotal, 'Used:', parsedUsed);

          if (parsedTotal > 0 && parsedUsed >= 0) {
            cachedVramTotal = parsedTotal;
            cachedVramUsed = parsedUsed;
            lastVramUpdateTime = Date.now();
            vramUsagePercent = (cachedVramUsed / cachedVramTotal) * 100;
            const logMsg = `[Main] VRAM from metrics: ${(cachedVramUsed / 1024 / 1024 / 1024).toFixed(2)} GB / ${(cachedVramTotal / 1024 / 1024 / 1024).toFixed(2)} GB (${vramUsagePercent.toFixed(1)}%)`;
            console.log(logMsg);
            sendLog('log-message', logMsg);
          }
        } else if (vramFreeMatch && vramTotalMatch) {
          const parsedTotal = Math.round(parseFloat(vramTotalMatch[1]));
          const vramFree = Math.round(parseFloat(vramFreeMatch[1]));
          const calculatedUsed = parsedTotal - vramFree;
          console.log('[Main] VRAM calculated - Total:', parsedTotal, 'Free:', vramFree, 'Used:', calculatedUsed);

          if (parsedTotal > 0 && calculatedUsed >= 0) {
            cachedVramTotal = parsedTotal;
            cachedVramUsed = calculatedUsed;
            lastVramUpdateTime = Date.now();
            vramUsagePercent = (cachedVramUsed / cachedVramTotal) * 100;
            const logMsg = `[Main] VRAM from metrics (calculated): ${(cachedVramUsed / 1024 / 1024 / 1024).toFixed(2)} GB / ${(cachedVramTotal / 1024 /
1024 / 1024).toFixed(2)} GB (${vramUsagePercent.toFixed(1)}%)`;
            console.log(logMsg);
            sendLog('log-message', logMsg);
          }
        } else {
          console.warn('[Main] VRAM metrics not found in /metrics response');
        }
      } catch (error) {
        const errorMsg = `[Main] Error fetching metrics: ${error.message}`;
        console.error(errorMsg);
        sendLog('log-message', errorMsg);
        if (currentModelConfig && currentModelConfig.gpuLayers > 0) {
          gpuUsage = 50;
        }
      }
      
          // 디버깅: 반환되는 값 확인
      console.log('[Main] Returning metrics - llamaServerProcess:', !!llamaServerProcess, 'vramTotal:', cachedVramTotal, 'vramUsed:', cachedVramUsed, 'vramUsage:', Math.round(vramUsagePercent), 'vramUsagePercent:', vramUsagePercent, 'lastUpdate:', lastVramUpdateTime > 0 ? new Date(lastVramUpdateTime).toISOString() : 'never');
      
      return {
        cpu: Math.round(cpuUsage),
        gpu: Math.round(gpuUsage), // GPU 사용량 (처리량 기반)
        memory: Math.round(memoryUsagePercent),
        totalMemory: totalMemory,
        usedMemory: usedMemory,
        freeMemory: freeMemory,
        vramTotal: cachedVramTotal, // VRAM 총량
        vramUsed: cachedVramUsed, // VRAM 사용량
        vramUsage: Math.round(vramUsagePercent) // VRAM 점유율 (%)
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
