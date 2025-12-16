const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// 인증 파일 경로
const AUTH_FILE_PATH = path.join(__dirname, '.auth.json');

// PBKDF2 기반 비밀번호 해시 생성
function derivePasswordHash(saltHex, password, iterations) {
  const salt = Buffer.from(saltHex, 'hex');
  return crypto.pbkdf2Sync(password, salt, iterations, 32, 'sha256').toString('hex');
}

// 인증 레코드 로드
function loadAuthFile() {
  try {
    if (!fs.existsSync(AUTH_FILE_PATH)) {
      return null;
    }
    const data = fs.readFileSync(AUTH_FILE_PATH, 'utf-8');
    return JSON.parse(data);
  } catch (error) {
    console.error('[Auth] Failed to load auth file:', error);
    return null;
  }
}

// 인증 레코드 저장
function saveAuthFile(record) {
  try {
    fs.writeFileSync(AUTH_FILE_PATH, JSON.stringify(record, null, 2), 'utf-8');
    return true;
  } catch (error) {
    console.error('[Auth] Failed to save auth file:', error);
    return false;
  }
}

// 토큰 생성
function generateToken() {
  return crypto.randomBytes(32).toString('hex');
}

// 토큰 저장소 (메모리 기반, 실제로는 Redis 등 사용 권장)
const activeTokens = new Set();

const server = http.createServer((req, res) => {
  // CORS 헤더 설정
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-LLM-UI-Auth');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = url.pathname;

  // /auth/status - 인증 상태 확인
  if (pathname === '/auth/status' && req.method === 'GET') {
    const authHeader = req.headers['x-llm-ui-auth'];
    const record = loadAuthFile();
    
    if (!record) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        initialized: false,
        authenticated: false,
        superAdminId: null
      }));
      return;
    }

    const authenticated = authHeader && activeTokens.has(authHeader);
    
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      initialized: true,
      authenticated,
      superAdminId: record.super_admin_id || null
    }));
    return;
  }

  // /auth/setup - 수퍼관리자 생성
  if (pathname === '/auth/setup' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const { superAdminId, password } = data;

        if (!superAdminId || !password) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'missing_fields' }));
          return;
        }

        // 이미 초기화되어 있으면 에러
        const existing = loadAuthFile();
        if (existing) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'already_initialized' }));
          return;
        }

        // 비밀번호 정책 검사 (최소 8자, 대소문자, 특수문자)
        if (password.length < 8 || !/[a-z]/.test(password) || !/[A-Z]/.test(password) || !/[^a-zA-Z0-9]/.test(password)) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'weak_password' }));
          return;
        }

        // 인증 레코드 생성
        const salt = crypto.randomBytes(16);
        const iterations = 100000;
        const hash = derivePasswordHash(salt.toString('hex'), password, iterations);

        const record = {
          super_admin_id: superAdminId,
          salt_hex: salt.toString('hex'),
          hash_hex: hash,
          iterations: iterations
        };

        if (!saveAuthFile(record)) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'failed_to_write_auth_file' }));
          return;
        }

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, superAdminId }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: 'invalid_json' }));
      }
    });
    return;
  }

  // /auth/login - 로그인
  if (pathname === '/auth/login' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const { superAdminId, password } = data;

        const record = loadAuthFile();
        if (!record) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'not_initialized' }));
          return;
        }

        if (superAdminId !== record.super_admin_id) {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'invalid_credentials' }));
          return;
        }

        const computedHash = derivePasswordHash(record.salt_hex, password, record.iterations);
        if (computedHash !== record.hash_hex) {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: 'invalid_credentials' }));
          return;
        }

        // 토큰 생성 및 저장
        const token = generateToken();
        activeTokens.add(token);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, token }));
      } catch (error) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: false, error: 'invalid_json' }));
      }
    });
    return;
  }

  // /auth/logout - 로그아웃
  if (pathname === '/auth/logout' && req.method === 'POST') {
    const authHeader = req.headers['x-llm-ui-auth'];
    if (authHeader) {
      activeTokens.delete(authHeader);
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  // 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'not_found' }));
});

const PORT = 8082; // 인증 서버는 8082 포트 사용

server.listen(PORT, () => {
  console.log(`[Auth Server] Started on port ${PORT}`);
});

// 서버 종료 처리
process.on('SIGTERM', () => {
  console.log('[Auth Server] Shutting down...');
  server.close(() => {
    console.log('[Auth Server] Stopped');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('[Auth Server] Shutting down...');
  server.close(() => {
    console.log('[Auth Server] Stopped');
    process.exit(0);
  });
});

module.exports = server;

