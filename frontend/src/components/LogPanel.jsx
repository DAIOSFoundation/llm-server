import React, { useState, useEffect, useRef } from 'react';
import PerformancePanel from './PerformancePanel';
import './LogPanel.css';
import { getActiveServerUrl } from '../services/api';

const LogPanel = () => {
  const [activeTab, setActiveTab] = useState('performance');
  const [logs, setLogs] = useState(['Waiting for server logs...']);
  const logContainerRef = useRef(null);

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

  // client-only: stream server logs via SSE (no polling)
  useEffect(() => {
    if (window.electronAPI) return;

    const token = (() => {
      try { return String(localStorage.getItem('llmServerUiAuthToken') || ''); } catch (_e) { return ''; }
    })();

    const ctrl = new AbortController();
    let stopped = false;

    const run = async () => {
      try {
        const serverUrl = getActiveServerUrl();
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
              if (line) setLogs(prev => [...prev, line]);
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
      stopped = true;
      ctrl.abort();
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
