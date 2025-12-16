const http = require('http');
const path = require('path');
const fs = require('fs');

// MLX 모델 검증을 위한 간단한 프록시 서버
// llama.cpp 서버와 함께 실행되어 /mlx-verify 엔드포인트 제공

const MLX_MODELS_DIR = path.join(__dirname, 'mlx', 'models');

function verifyMlxModel(modelId) {
  if (!modelId || typeof modelId !== 'string') {
    return { exists: false, error: 'Invalid modelId' };
  }

  const modelDir = path.join(MLX_MODELS_DIR, modelId);
  
  console.log(`[MLX Verify Proxy] Checking model: ${modelId}`);
  console.log(`[MLX Verify Proxy] Model dir: ${modelDir}`);
  console.log(`[MLX Verify Proxy] Exists: ${fs.existsSync(modelDir)}`);

  if (fs.existsSync(modelDir)) {
    const stats = fs.statSync(modelDir);
    if (stats.isDirectory()) {
      const configPath = path.join(modelDir, 'config.json');
      const hasConfig = fs.existsSync(configPath);
      console.log(`[MLX Verify Proxy] Config exists: ${hasConfig}`);
      
      if (hasConfig) {
        return { exists: true, path: modelDir };
      } else {
        return { exists: false, error: 'config.json not found' };
      }
    } else {
      return { exists: false, error: 'Model path is not a directory' };
    }
  }

  return { exists: false, error: 'Model directory not found' };
}

// 간단한 HTTP 서버로 /mlx-verify 엔드포인트 제공
const proxyServer = http.createServer((req, res) => {
  // CORS 헤더 설정
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  if (req.url === '/mlx-verify' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      try {
        const payload = JSON.parse(body);
        const result = verifyMlxModel(payload.model);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (error) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ exists: false, error: error.message }));
      }
    });
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not found' }));
  }
});

// 포트 8084에서 실행
const PORT = 8084;
proxyServer.listen(PORT, () => {
  console.log(`[MLX Verify Proxy] Server running on port ${PORT}`);
});

module.exports = { verifyMlxModel };

