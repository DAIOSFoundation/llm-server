import React, { useState, useEffect, useRef } from 'react';
import PerformancePanel from './PerformancePanel';
import './LogPanel.css';
import { getActiveServerUrl, getActiveModelFormat } from '../services/api';

const LogPanel = () => {
  const [activeTab, setActiveTab] = useState('performance');
  const [logs, setLogs] = useState(['Waiting for server logs...']);
  const logContainerRef = useRef(null);
  const websocketRef = useRef(null);

  useEffect(() => {
    const handleLogMessage = (message) => {
      setLogs(prevLogs => [...prevLogs, message]);
    };

    const handleClientLogEvent = (event) => {
      if (event && event.detail) {
        setLogs(prevLogs => [...prevLogs, event.detail]);
      }
    };

    if (window.electronAPI) {
      window.electronAPI.onLogMessage(handleLogMessage);
    }

    // 프론트엔드에서 발생시키는 server-log 이벤트도 함께 수신
    window.addEventListener('server-log', handleClientLogEvent);

    return () => {
      if (window.electronAPI) {
        window.electronAPI.removeLogListener();
      }
      window.removeEventListener('server-log', handleClientLogEvent);
    };
  }, []);

  // client-only: stream server logs via SSE (GGUF) or WebSocket (MLX)
  useEffect(() => {
    if (window.electronAPI) return;

    const modelFormat = getActiveModelFormat();
    const useWebSocket = modelFormat === 'mlx';

    if (useWebSocket && typeof WebSocket === 'undefined') return;

    let stopped = false;

    const connect = () => {
      if (stopped) return;

      const serverUrl = getActiveServerUrl();

      if (useWebSocket) {
        // MLX 모델: WebSocket 사용
        // 서버가 준비되었는지 먼저 확인
        const checkAndConnect = async () => {
          try {
            const healthResponse = await fetch(`${serverUrl}/health`, { signal: AbortSignal.timeout(2000) });
            if (healthResponse.ok) {
              const healthData = await healthResponse.json();
              // 서버가 ready 상태이거나 loading 상태일 때만 연결 시도
              if (healthData.status === 'ready' || healthData.status === 'loading') {
                const wsUrl = serverUrl.replace('http://', 'ws://').replace('https://', 'wss://');
                const ws = new WebSocket(`${wsUrl}/logs/stream`);
                websocketRef.current = ws;

                ws.onopen = () => {
                  // WebSocket 연결 성공
                };

                ws.onmessage = (event) => {
                  try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'log' && data.text) {
                      const line = String(data.text).trimEnd();
                      if (line) {
                        setLogs(prev => [...prev, line]);
                        // server-log 이벤트 브로드캐스트 (Header에서 프로그레스 파싱용)
                        window.dispatchEvent(new CustomEvent('server-log', { detail: line }));
                      }
                    } else if (data.type === 'progress' && data.text) {
                      // 프로그레스 정보는 로그 패널에 표시하지 않고 Header로만 전달
                      window.dispatchEvent(new CustomEvent('server-log', { detail: data.text }));
                    }
                  } catch (_e) {
                    // ignore
                  }
                };

                ws.onerror = () => {
                  // 에러는 조용히 처리, onclose에서 재연결
                };

                ws.onclose = () => {
                  if (!stopped) {
                    setTimeout(connect, 5000);  // 재연결 간격 증가
                  }
                };
              } else {
                // 서버가 준비되지 않았으면 재시도
                if (!stopped) {
                  setTimeout(connect, 5000);
                }
              }
            } else {
              // 헬스 체크 실패 시 재시도
              if (!stopped) {
                setTimeout(connect, 5000);
              }
            }
          } catch (e) {
            // 서버가 아직 시작되지 않았거나 연결 불가 - 재시도
            if (!stopped) {
              setTimeout(connect, 5000);
            }
          }
        };
        
        checkAndConnect();
      } else {
        // GGUF 모델: SSE 사용
        const token = (() => {
          try { return String(localStorage.getItem('llmServerUiAuthToken') || ''); } catch (_e) { return ''; }
        })();

        const ctrl = new AbortController();

        const run = async () => {
          try {
            const res = await fetch(`${serverUrl}/logs/stream`, {
              method: 'GET',
              headers: token ? { 'X-LLM-UI-Auth': token } : {},
              signal: ctrl.signal,
            });

            if (!res.ok || !res.body) {
              return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buf = '';
            let eventName = '';
            let dataLines = [];

            const flush = () => {
              if (eventName === 'log' && dataLines.length > 0) {
                const data = dataLines.join('\n');
                try {
                  const json = JSON.parse(data);
                  const line = String(json.text || '').trimEnd();
                  if (line) {
                    setLogs(prev => [...prev, line]);
                    // server-log 이벤트 브로드캐스트 (Header에서 프로그레스 파싱용)
                    window.dispatchEvent(new CustomEvent('server-log', { detail: line }));
                  }
                } catch (_e) {
                  // ignore
                }
              }
              eventName = '';
              dataLines = [];
            };

            while (!stopped) {
              const { value, done } = await reader.read();
              if (done) break;
              buf += decoder.decode(value, { stream: true });

              // parse SSE by lines
              let idx;
              while ((idx = buf.indexOf('\n')) >= 0) {
                const rawLine = buf.slice(0, idx);
                buf = buf.slice(idx + 1);
                const line = rawLine.replace(/\r$/, '');

                if (line === '') {
                  flush();
                  continue;
                }
                if (line.startsWith(':')) {
                  continue; // comment
                }
                if (line.startsWith('event:')) {
                  eventName = line.slice('event:'.length).trim();
                  continue;
                }
                if (line.startsWith('data:')) {
                  dataLines.push(line.slice('data:'.length).trimStart());
                }
              }
            }
          } catch (_e) {
            // ignore
          }
        };

        run();

        return () => {
          ctrl.abort();
        };
      }
    };

    connect();

    return () => {
      stopped = true;
      if (websocketRef.current) {
        try { websocketRef.current.close(); } catch (_e) {}
        websocketRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="log-panel">
      <div className="tab-header">
        <button 
          className={`tab-button ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          Performance
        </button>
        <button 
          className={`tab-button ${activeTab === 'logs' ? 'active' : ''}`}
          onClick={() => setActiveTab('logs')}
        >
          Server Log
        </button>
      </div>
      <div className="tab-content">
        {activeTab === 'performance' ? (
          <PerformancePanel />
        ) : (
          <div className="log-container" ref={logContainerRef}>
            {logs.map((log, index) => (
              <pre key={index} className="log-line">{log}</pre>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LogPanel;
