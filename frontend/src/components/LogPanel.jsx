import React, { useState, useEffect, useRef } from 'react';
import PerformancePanel from './PerformancePanel';
import './LogPanel.css';

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
