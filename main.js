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

// ----------------------------
// GGUF metadata parser (no full-file read)
// ----------------------------
const GGUF_VALUE_TYPE = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
};

const GGML_TYPE_NAME = {
  0: 'F32',
  1: 'F16',
  2: 'Q4_0',
  3: 'Q4_1',
  6: 'Q5_0',
  7: 'Q5_1',
  8: 'Q8_0',
  9: 'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'I8',
  25: 'I16',
  26: 'I32',
  27: 'I64',
  28: 'F64',
  29: 'IQ1_M',
  30: 'BF16',
  34: 'TQ1_0',
  35: 'TQ2_0',
  39: 'MXFP4',
};

// subset of llama_ftype values (llama.cpp/include/llama.h)
const LLAMA_FTYPE_NAME = {
  0: 'ALL_F32',
  1: 'MOSTLY_F16',
  2: 'MOSTLY_Q4_0',
  3: 'MOSTLY_Q4_1',
  7: 'MOSTLY_Q8_0',
  8: 'MOSTLY_Q5_0',
  9: 'MOSTLY_Q5_1',
  10: 'MOSTLY_Q2_K',
  11: 'MOSTLY_Q3_K_S',
  12: 'MOSTLY_Q3_K_M',
  13: 'MOSTLY_Q3_K_L',
  14: 'MOSTLY_Q4_K_S',
  15: 'MOSTLY_Q4_K_M',
  16: 'MOSTLY_Q5_K_S',
  17: 'MOSTLY_Q5_K_M',
  18: 'MOSTLY_Q6_K',
  19: 'MOSTLY_IQ2_XXS',
  20: 'MOSTLY_IQ2_XS',
  21: 'MOSTLY_Q2_K_S',
  22: 'MOSTLY_IQ3_XS',
  23: 'MOSTLY_IQ3_XXS',
  24: 'MOSTLY_IQ1_S',
  25: 'MOSTLY_IQ4_NL',
  26: 'MOSTLY_IQ3_S',
  27: 'MOSTLY_IQ3_M',
  28: 'MOSTLY_IQ2_S',
  29: 'MOSTLY_IQ2_M',
  30: 'MOSTLY_IQ4_XS',
  31: 'MOSTLY_IQ1_M',
  32: 'MOSTLY_BF16',
  36: 'MOSTLY_TQ1_0',
  37: 'MOSTLY_TQ2_0',
  38: 'MOSTLY_MXFP4_MOE',
};

function parseGgufInfoFromFile(filePath) {
  const fd = fs.openSync(filePath, 'r');
  let pos = 0;

  const MAX_STRING_BYTES = 4 * 1024 * 1024; // 4MB safeguard

  const readBytes = (n) => {
    const buf = Buffer.allocUnsafe(n);
    const read = fs.readSync(fd, buf, 0, n, pos);
    if (read !== n) {
      throw new Error(`Unexpected EOF while reading ${n} bytes`);
    }
    pos += n;
    return buf;
  };

  const readU32 = () => readBytes(4).readUInt32LE(0);
  const readI32 = () => readBytes(4).readInt32LE(0);
  const readI64 = () => {
    const v = readBytes(8).readBigInt64LE(0);
    const n = Number(v);
    if (!Number.isSafeInteger(n)) {
      throw new Error('int64 value is not safe integer');
    }
    return n;
  };
  const readU64 = () => {
    const v = readBytes(8).readBigUInt64LE(0);
    const n = Number(v);
    if (!Number.isSafeInteger(n)) {
      throw new Error('uint64 value is not safe integer');
    }
    return n;
  };
  const readString = () => {
    const len = readU64();
    if (len < 0 || len > MAX_STRING_BYTES) {
      throw new Error(`String length out of range: ${len}`);
    }
    return readBytes(len).toString('utf8');
  };

  const skipBytes = (n) => {
    pos += n;
  };

  const sizeOfGgufType = (t) => {
    switch (t) {
      case GGUF_VALUE_TYPE.UINT8:
      case GGUF_VALUE_TYPE.INT8:
      case GGUF_VALUE_TYPE.BOOL:
        return 1;
      case GGUF_VALUE_TYPE.UINT16:
      case GGUF_VALUE_TYPE.INT16:
        return 2;
      case GGUF_VALUE_TYPE.UINT32:
      case GGUF_VALUE_TYPE.INT32:
      case GGUF_VALUE_TYPE.FLOAT32:
        return 4;
      case GGUF_VALUE_TYPE.UINT64:
      case GGUF_VALUE_TYPE.INT64:
      case GGUF_VALUE_TYPE.FLOAT64:
        return 8;
      default:
        return null;
    }
  };

  const skipValueByType = (t) => {
    if (t === GGUF_VALUE_TYPE.STRING) {
      // string = u64 len + bytes
      const len = readU64();
      if (len < 0 || len > MAX_STRING_BYTES) {
        throw new Error(`String length out of range: ${len}`);
      }
      skipBytes(len);
      return;
    }

    if (t === GGUF_VALUE_TYPE.ARRAY) {
      const elemType = readI32();
      const n = readU64();
      if (elemType === GGUF_VALUE_TYPE.STRING) {
        for (let i = 0; i < n; i++) {
          const len = readU64();
          if (len < 0 || len > MAX_STRING_BYTES) {
            throw new Error(`String length out of range: ${len}`);
          }
          skipBytes(len);
        }
        return;
      }
      const elemSize = sizeOfGgufType(elemType);
      if (elemSize == null) {
        throw new Error(`Unsupported array element type: ${elemType}`);
      }
      skipBytes(n * elemSize);
      return;
    }

    const sz = sizeOfGgufType(t);
    if (sz == null) {
      throw new Error(`Unsupported GGUF value type: ${t}`);
    }
    skipBytes(sz);
  };

  try {
    const magic = readBytes(4).toString('ascii');
    if (magic !== 'GGUF') {
      throw new Error(`Not a GGUF file (magic=${magic})`);
    }

    const version = readU32();
    // GGUF spec: n_tensors, n_kv are uint64
    const nTensors = readU64();
    const nKv = readU64();

    let fileTypeId = null;
    const kvKeysRead = [];

    for (let i = 0; i < nKv; i++) {
      const key = readString();
      const valueType = readI32();

      if (kvKeysRead.length < 64) {
        kvKeysRead.push(key);
      }

      if (key === 'general.file_type' && (valueType === GGUF_VALUE_TYPE.UINT32 || valueType === GGUF_VALUE_TYPE.INT32)) {
        fileTypeId = valueType === GGUF_VALUE_TYPE.UINT32 ? readU32() : readI32();
      } else {
        skipValueByType(valueType);
      }
    }

    const typeCounts = {};
    const qkv = { q: null, k: null, v: null };

    const matchQ = (name) => /(attn_q|q_proj|wq|query)/i.test(name);
    const matchK = (name) => /(attn_k|k_proj|wk|key)/i.test(name);
    const matchV = (name) => /(attn_v|v_proj|wv|value)/i.test(name);

    for (let i = 0; i < nTensors; i++) {
      const tName = readString();
      const nDims = readU32();
      for (let d = 0; d < nDims; d++) {
        readU64(); // dim (uint64)
      }
      const tType = readI32(); // ggml_type
      readU64(); // offset

      const typeName = GGML_TYPE_NAME[tType] || `TYPE_${tType}`;
      typeCounts[typeName] = (typeCounts[typeName] || 0) + 1;

      // record first-seen Q/K/V weight tensor type
      if (!qkv.q && matchQ(tName)) qkv.q = typeName;
      if (!qkv.k && matchK(tName)) qkv.k = typeName;
      if (!qkv.v && matchV(tName)) qkv.v = typeName;
    }

    const fileTypeName = fileTypeId != null ? (LLAMA_FTYPE_NAME[fileTypeId & ~1024] || `FTYPE_${fileTypeId}`) : null;

    return {
      ok: true,
      filePath,
      ggufVersion: version,
      fileTypeId,
      fileTypeName,
      tensorTypes: typeCounts,
      qkv,
      kvKeysSample: kvKeysRead,
    };
  } finally {
    fs.closeSync(fd);
  }
}

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

  ipcMain.handle('get-gguf-info', async (_event, modelPath) => {
    try {
      if (!modelPath || typeof modelPath !== 'string') {
        return { ok: false, error: 'Invalid modelPath' };
      }
      if (!fs.existsSync(modelPath)) {
        return { ok: false, error: 'File not found' };
      }
      return parseGgufInfoFromFile(modelPath);
    } catch (error) {
      return { ok: false, error: error.message || String(error) };
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
