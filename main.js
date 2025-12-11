const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');
const os = require('os');

const configPath = path.join(app.getPath('userData'), 'config.json');
let llamaServerProcess = null;
let mainWindow = null;
let cachedVramTotal = 0; // VRAM 용량 캐싱

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
      // GPU 사용량 (VRAM이 감지되었으면, 메모리 사용률을 GPU 부하의 대략적인 지표로 활용하거나 별도 로직 필요)
      // 현재는 VRAM Total만 정확하므로, GPU Load는 임시 랜덤값 유지 또는 추후 llama-server 상태 연동 필요
      const gpuUsage = Math.random() * 100;
      
      return {
        cpu: Math.round(cpuUsage),
        gpu: Math.round(gpuUsage),
        memory: Math.round(memoryUsagePercent),
        totalMemory: totalMemory,
        usedMemory: usedMemory,
        freeMemory: freeMemory,
        vramTotal: cachedVramTotal // VRAM 정보 추가
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
