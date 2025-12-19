import React, { useMemo, useState, useEffect, useRef } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { LLAMA_BASE_URL } from '../services/api';
import './GuidePage.css';

const getExampleModelId = () => {
  try {
    const raw = localStorage.getItem('llmServerClientConfig');
    if (raw) {
      const cfg = JSON.parse(raw);
      const models = cfg?.models || [];
      const active = models.find((m) => m?.id === cfg?.activeModelId);
      const modelPath = String(active?.modelPath || '').trim();
      if (modelPath) return modelPath;
    }
  } catch (_e) {}
  try {
    const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
    const modelPath = String(config.modelPath || '').trim();
    if (modelPath) return modelPath;
  } catch (_e) {}
  return 'llama31-banyaa-q4_k_m';
};

const CodeCard = ({ id, title, description, code, isOpen, onToggle }) => {
  return (
    <section id={id} className="guide-card">
      <div className="guide-card-header">
        <div className="guide-card-title-row">
          <h3>{title}</h3>
          <button
            type="button"
            className="guide-toggle-button"
            onClick={onToggle}
            aria-expanded={isOpen}
            aria-label={isOpen ? 'collapse' : 'expand'}
            title={isOpen ? '접기' : '펼치기'}
          >
            <span className={`guide-chevron ${isOpen ? 'open' : ''}`}>▾</span>
          </button>
        </div>
        {description ? <p>{description}</p> : null}
      </div>
      {isOpen ? (
        <pre className="guide-code">
          <code>{code}</code>
        </pre>
      ) : null}
    </section>
  );
};

const GuidePage = () => {
  const { t } = useLanguage();

  const modelId = useMemo(() => getExampleModelId(), []);

  const [activeSection, setActiveSection] = useState('gguf'); // 'gguf', 'mlx', 'auth', 'client'

  const apiSections = useMemo(() => {
    const base = LLAMA_BASE_URL;
    const mlxBase = base.replace(':8080', ':8081');
    const authBase = base.replace(':8080', ':8082');
    const clientBase = base.replace(':8080', ':8083');
    const sampleModel = modelId;

    return {
      gguf: {
        title: 'GGUF Server (Port 8080)',
        description: 'llama.cpp server APIs for GGUF format models',
        cards: [
          {
            key: 'gguf-health',
            title: 'Health Check',
            description: 'Check server health status',
            code: `# curl
curl -sS "${base}/health"

# JavaScript
const res = await fetch("${base}/health");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${base}/health")
print(r.json())`,
          },
          {
            key: 'gguf-models',
            title: 'List Models',
            description: 'Get list of available models (Router Mode)',
            code: `# curl
curl -sS "${base}/models"

# JavaScript
const res = await fetch("${base}/models");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${base}/models")
print(r.json())`,
          },
          {
            key: 'gguf-load-model',
            title: 'Load Model',
            description: 'Load a model (Router Mode)',
            code: `# curl
curl -X POST "${base}/models/load" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "${sampleModel}"}'

# JavaScript
const res = await fetch("${base}/models/load", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: "${sampleModel}" })
});
const data = await res.json();

# Python
import requests
r = requests.post("${base}/models/load", 
  json={"name": "${sampleModel}"})
print(r.json())`,
          },
          {
            key: 'gguf-completion',
            title: 'Completion (SSE Streaming)',
            description: 'Generate text completion with Server-Sent Events streaming',
            code: `# curl (SSE streaming)
curl -N -X POST "${base}/completion" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Hello, world",
    "stream": true,
    "n_predict": 128,
    "temperature": 0.7
  }'

# JavaScript (SSE streaming)
const res = await fetch("${base}/completion", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Hello, world",
    stream: true,
    n_predict: 128,
    temperature: 0.7
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  console.log(chunk);
}

# Python (SSE streaming)
import requests
import json

r = requests.post("${base}/completion",
  json={
    "prompt": "Hello, world",
    "stream": True,
    "n_predict": 128,
    "temperature": 0.7
  },
  stream=True
)

for line in r.iter_lines():
  if line:
    print(line.decode('utf-8'))`,
          },
          {
            key: 'gguf-metrics',
            title: 'Get Metrics',
            description: 'Get server performance metrics (Prometheus format)',
            code: `# curl
curl -sS "${base}/metrics"

# JavaScript
const res = await fetch("${base}/metrics");
const text = await res.text();
console.log(text);

# Python
import requests
r = requests.get("${base}/metrics")
print(r.text)`,
          },
          {
            key: 'gguf-metrics-stream',
            title: 'Metrics Stream (SSE)',
            description: 'Real-time metrics streaming via Server-Sent Events',
            code: `# curl (SSE streaming)
curl -N "${base}/metrics/stream"

# JavaScript (SSE streaming)
const eventSource = new EventSource("${base}/metrics/stream");
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Metrics:", data);
};

# Python (SSE streaming)
import requests
import json

r = requests.get("${base}/metrics/stream", stream=True)
for line in r.iter_lines():
  if line.startswith(b'data: '):
    data = json.loads(line[6:].decode('utf-8'))
    print("Metrics:", data)`,
          },
          {
            key: 'gguf-tokenize',
            title: 'Tokenize',
            description: 'Convert text to tokens',
            code: `# curl
curl -X POST "${base}/tokenize" \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Hello, world"}'

# JavaScript
const res = await fetch("${base}/tokenize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ content: "Hello, world" })
});
const data = await res.json();
console.log(data.tokens);

# Python
import requests
r = requests.post("${base}/tokenize",
  json={"content": "Hello, world"})
print(r.json()["tokens"])`,
          },
          {
            key: 'gguf-gguf-info',
            title: 'GGUF Info',
            description: 'Get GGUF file metadata (quantization, tensor types, etc.)',
            code: `# curl
curl -X POST "${base}/gguf-info" \\
  -H "Content-Type: application/json" \\
  -d '{"path": "/path/to/model.gguf"}'

# JavaScript
const res = await fetch("${base}/gguf-info", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ path: "/path/to/model.gguf" })
});
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.post("${base}/gguf-info",
  json={"path": "/path/to/model.gguf"})
print(r.json())`,
          },
          {
            key: 'gguf-logs-stream',
            title: 'Logs Stream (SSE)',
            description: 'Real-time server logs streaming via Server-Sent Events',
            code: `# curl (SSE streaming)
curl -N "${base}/logs/stream"

# JavaScript (SSE streaming)
const eventSource = new EventSource("${base}/logs/stream");
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Log:", data.text);
};

# Python (SSE streaming)
import requests
import json

r = requests.get("${base}/logs/stream", stream=True)
for line in r.iter_lines():
  if line.startswith(b'data: '):
    data = json.loads(line[6:].decode('utf-8'))
    print("Log:", data.get("text", ""))`,
          },
        ],
      },
      mlx: {
        title: 'MLX Server (Port 8081)',
        description: 'Python FastAPI server APIs for MLX format models',
        cards: [
          {
            key: 'mlx-health',
            title: 'Health Check',
            description: 'Check server health status',
            code: `# curl
curl -sS "${mlxBase}/health"

# JavaScript
const res = await fetch("${mlxBase}/health");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${mlxBase}/health")
print(r.json())`,
          },
          {
            key: 'mlx-models',
            title: 'List Models',
            description: 'Get list of available models',
            code: `# curl
curl -sS "${mlxBase}/models"

# JavaScript
const res = await fetch("${mlxBase}/models");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${mlxBase}/models")
print(r.json())`,
          },
          {
            key: 'mlx-chat-http',
            title: 'Chat (HTTP POST with SSE)',
            description: 'Chat completion via HTTP POST with Server-Sent Events streaming',
            code: `# curl (SSE streaming)
curl -N -X POST "${mlxBase}/chat" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Hello, world",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.1
  }'

# JavaScript (SSE streaming)
const res = await fetch("${mlxBase}/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Hello, world",
    max_tokens: 128,
    temperature: 0.7,
    top_p: 0.95,
    repeat_penalty: 1.1
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  console.log(chunk);
}

# Python (SSE streaming)
import requests
import json

r = requests.post("${mlxBase}/chat",
  json={
    "prompt": "Hello, world",
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.1
  },
  stream=True
)

for line in r.iter_lines():
  if line:
    print(line.decode('utf-8'))`,
          },
          {
            key: 'mlx-chat-websocket',
            title: 'Chat (WebSocket)',
            description: 'Real-time chat completion via WebSocket with token-by-token streaming',
            code: `# JavaScript (WebSocket)
const ws = new WebSocket("ws://localhost:8081/chat/ws");

ws.onopen = () => {
  ws.send(JSON.stringify({
    prompt: "Hello, world",
    max_tokens: 128,
    temperature: 0.7,
    top_p: 0.95,
    repeat_penalty: 1.1
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "token") {
    process.stdout.write(data.content);
  } else if (data.type === "done") {
    console.log("\\nGeneration complete");
    ws.close();
  } else if (data.type === "error") {
    console.error("Error:", data.message);
    ws.close();
  }
};

# Python (WebSocket)
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8081/chat/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "prompt": "Hello, world",
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.95,
            "repeat_penalty": 1.1
        }))
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if data["type"] == "token":
                print(data["content"], end="", flush=True)
            elif data["type"] == "done":
                print("\\nGeneration complete")
                break
            elif data["type"] == "error":
                print("Error:", data["message"])
                break

asyncio.run(chat())`,
          },
          {
            key: 'mlx-completion',
            title: 'Completion (llama.cpp compatible)',
            description: 'Completion endpoint compatible with llama.cpp API (SSE streaming)',
            code: `# curl (SSE streaming)
curl -N -X POST "${mlxBase}/completion" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Hello, world",
    "stream": true,
    "n_predict": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.1
  }'

# JavaScript (SSE streaming)
const res = await fetch("${mlxBase}/completion", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "Hello, world",
    stream: true,
    n_predict: 128,
    temperature: 0.7,
    top_p: 0.95,
    repeat_penalty: 1.1
  })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  console.log(chunk);
}

# Python (SSE streaming)
import requests
import json

r = requests.post("${mlxBase}/completion",
  json={
    "prompt": "Hello, world",
    "stream": True,
    "n_predict": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "repeat_penalty": 1.1
  },
  stream=True
)

for line in r.iter_lines():
  if line:
    print(line.decode('utf-8'))`,
          },
          {
            key: 'mlx-metrics',
            title: 'Get Metrics',
            description: 'Get server performance metrics (VRAM, CPU, token speed)',
            code: `# curl
curl -sS "${mlxBase}/metrics"

# JavaScript
const res = await fetch("${mlxBase}/metrics");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${mlxBase}/metrics")
print(r.json())`,
          },
          {
            key: 'mlx-metrics-websocket',
            title: 'Metrics Stream (WebSocket)',
            description: 'Real-time metrics streaming via WebSocket',
            code: `# JavaScript (WebSocket)
const ws = new WebSocket("ws://localhost:8081/metrics/stream");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Metrics:", data);
  // data contains: vramTotal, vramUsed, sysMemTotal, sysMemUsed, 
  // cpuCores, procCpuSec, tps, predictedTotal
};

# Python (WebSocket)
import asyncio
import websockets
import json

async def metrics():
    uri = "ws://localhost:8081/metrics/stream"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print("Metrics:", data)

asyncio.run(metrics())`,
          },
          {
            key: 'mlx-tokenize',
            title: 'Tokenize',
            description: 'Convert text to tokens',
            code: `# curl
curl -X POST "${mlxBase}/tokenize" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello, world"}'

# JavaScript
const res = await fetch("${mlxBase}/tokenize", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Hello, world" })
});
const data = await res.json();
console.log(data.tokens);

# Python
import requests
r = requests.post("${mlxBase}/tokenize",
  json={"text": "Hello, world"})
print(r.json()["tokens"])`,
          },
          {
            key: 'mlx-logs-websocket',
            title: 'Logs Stream (WebSocket)',
            description: 'Real-time server logs streaming via WebSocket',
            code: `# JavaScript (WebSocket)
const ws = new WebSocket("ws://localhost:8081/logs/stream");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "log") {
    console.log("Log:", data.text);
  }
};

# Python (WebSocket)
import asyncio
import websockets
import json

async def logs():
    uri = "ws://localhost:8081/logs/stream"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get("type") == "log":
                print("Log:", data.get("text", ""))

asyncio.run(logs())`,
          },
        ],
      },
      auth: {
        title: 'Authentication Server (Port 8082)',
        description: 'User authentication APIs (login, logout, setup)',
        cards: [
          {
            key: 'auth-status',
            title: 'Get Status',
            description: 'Check authentication server status',
            code: `# curl
curl -sS "${authBase}/auth/status"

# JavaScript
const res = await fetch("${authBase}/auth/status");
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.get("${authBase}/auth/status")
print(r.json())`,
          },
          {
            key: 'auth-setup',
            title: 'Setup (Create Super Admin)',
            description: 'Create super admin account (first run only)',
            code: `# curl
curl -X POST "${authBase}/auth/setup" \\
  -H "Content-Type: application/json" \\
  -d '{"password": "your-secure-password"}'

# JavaScript
const res = await fetch("${authBase}/auth/setup", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ password: "your-secure-password" })
});
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.post("${authBase}/auth/setup",
  json={"password": "your-secure-password"})
print(r.json())`,
          },
          {
            key: 'auth-login',
            title: 'Login',
            description: 'Login with super admin credentials',
            code: `# curl
curl -X POST "${authBase}/auth/login" \\
  -H "Content-Type: application/json" \\
  -d '{"password": "your-secure-password"}' \\
  -c cookies.txt

# JavaScript
const res = await fetch("${authBase}/auth/login", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  credentials: "include",
  body: JSON.stringify({ password: "your-secure-password" })
});
const data = await res.json();
console.log(data);

# Python (with session)
import requests

session = requests.Session()
r = session.post("${authBase}/auth/login",
  json={"password": "your-secure-password"})
print(r.json())
# Session cookie is automatically stored for subsequent requests`,
          },
          {
            key: 'auth-logout',
            title: 'Logout',
            description: 'Logout and clear session',
            code: `# curl
curl -X POST "${authBase}/auth/logout" \\
  -b cookies.txt

# JavaScript
const res = await fetch("${authBase}/auth/logout", {
  method: "POST",
  credentials: "include"
});
const data = await res.json();
console.log(data);

# Python (with session)
import requests

session = requests.Session()
# ... login first ...
r = session.post("${authBase}/auth/logout")
print(r.json())`,
          },
        ],
      },
      client: {
        title: 'Client Server Manager (Port 8083)',
        description: 'Client server manager APIs for model configuration',
        cards: [
          {
            key: 'client-save-config',
            title: 'Save Config',
            description: 'Save model configuration and automatically start servers',
            code: `# curl
curl -X POST "${clientBase}/api/save-config" \\
  -H "Content-Type: application/json" \\
  -d '{
    "models": [
      {
        "id": "model-1",
        "name": "Model 1",
        "modelPath": "path/to/model",
        "format": "gguf"
      }
    ],
    "activeModelId": "model-1"
  }'

# JavaScript
const res = await fetch("${clientBase}/api/save-config", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    models: [
      {
        id: "model-1",
        name: "Model 1",
        modelPath: "path/to/model",
        format: "gguf"
      }
    ],
    activeModelId: "model-1"
  })
});
const data = await res.json();
console.log(data);

# Python
import requests
r = requests.post("${clientBase}/api/save-config",
  json={
    "models": [
      {
        "id": "model-1",
        "name": "Model 1",
        "modelPath": "path/to/model",
        "format": "gguf"
      }
    ],
    "activeModelId": "model-1"
  })
print(r.json())`,
          },
        ],
      },
    };
  }, [modelId]);

  const currentSection = apiSections[activeSection] || apiSections.gguf;
  const [activeAnchor, setActiveAnchor] = useState(null);
  const contentRef = useRef(null);

  // default: collapsed
  const [openCards, setOpenCards] = useState(() => ({}));

  const isCardOpen = (key) => Boolean(openCards[key]);
  const toggleCard = (key) => {
    setOpenCards((prev) => ({ ...(prev || {}), [key]: !prev?.[key] }));
  };

  // Scroll to anchor
  const scrollToAnchor = (anchorId) => {
    const element = document.getElementById(anchorId);
    if (element) {
      const offset = 80; // Offset for fixed header
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
      setActiveAnchor(anchorId);
    }
  };

  // Track active anchor on scroll
  useEffect(() => {
    const handleScroll = () => {
      const cards = currentSection.cards || [];
      const scrollPosition = window.scrollY + 100;

      for (let i = cards.length - 1; i >= 0; i--) {
        const card = document.getElementById(cards[i].key);
        if (card) {
          const cardTop = card.offsetTop;
          if (scrollPosition >= cardTop) {
            setActiveAnchor(cards[i].key);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial check

    return () => window.removeEventListener('scroll', handleScroll);
  }, [currentSection, activeSection]);

  // Generate table of contents
  const tableOfContents = useMemo(() => {
    return currentSection.cards.map((card) => ({
      id: card.key,
      title: card.title,
    }));
  }, [currentSection]);

  return (
    <div className="guide-page">
      {/* Sidebar */}
      <aside className="guide-sidebar">
        <div className="guide-sidebar-content">
          <h3 className="guide-sidebar-title">Table of Contents</h3>
          <nav className="guide-sidebar-nav">
            <ul className="guide-toc-list">
              {tableOfContents.map((item) => (
                <li key={item.id} className="guide-toc-item">
                  <a
                    href={`#${item.id}`}
                    className={`guide-toc-link ${activeAnchor === item.id ? 'active' : ''}`}
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToAnchor(item.id);
                    }}
                  >
                    {item.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <div className="guide-content" ref={contentRef}>
        <div className="guide-header">
          <h2>{t('guide.title')}</h2>
          <p>{t('guide.subtitle')}</p>
          <div className="guide-meta">
            <div>
              <span className="guide-meta-label">{t('guide.baseUrlLabel')}</span>
              <code className="guide-inline-code">{LLAMA_BASE_URL}</code>
            </div>
            <div>
              <span className="guide-meta-label">{t('guide.exampleModelId')}</span>
              <code className="guide-inline-code">{modelId}</code>
            </div>
          </div>
          <div className="guide-note">{t('guide.routerNote')}</div>
        </div>

        {/* Section Tabs */}
        <div className="guide-section-tabs">
          {Object.entries(apiSections).map(([key, section]) => (
            <button
              key={key}
              className={`guide-section-tab ${activeSection === key ? 'active' : ''}`}
              onClick={() => {
                setActiveSection(key);
                setActiveAnchor(null);
                // Scroll to top of content
                if (contentRef.current) {
                  contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
            >
              {section.title}
            </button>
          ))}
        </div>

        {/* Section Description */}
        <div className="guide-section-description">
          <p>{currentSection.description}</p>
        </div>

        {/* API Cards */}
        <div className="guide-grid">
          {currentSection.cards.map((c) => (
            <CodeCard
              key={c.key}
              id={c.key}
              title={c.title}
              description={c.description}
              code={c.code}
              isOpen={isCardOpen(c.key)}
              onToggle={() => toggleCard(c.key)}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default GuidePage;
