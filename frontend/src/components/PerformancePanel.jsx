import React, { useState, useEffect, useRef } from 'react';
import './PerformancePanel.css';
import TokenDebugPanel from './TokenDebugPanel';
import { LLAMA_BASE_URL } from '../services/api';

const PerformancePanel = () => {
  const [cpuUsage, setCpuUsage] = useState(0);
  const [gpuUsage, setGpuUsage] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [vramTotal, setVramTotal] = useState(0); // VRAM 총량
  const [vramUsed, setVramUsed] = useState(0); // VRAM 사용량
  const [contextUsage, setContextUsage] = useState(0);
  const [contextUsed, setContextUsed] = useState(0);
  const [contextSize, setContextSize] = useState(2048);
  const [tokenSpeed, setTokenSpeed] = useState(0);
  const [tokenCount, setTokenCount] = useState(0);
  const tokenSpeedRef = useRef(0);
  const lastTokenTimeRef = useRef(Date.now());
  const tokenCountRef = useRef(0);
  const contextSizeRef = useRef(2048);
  const lastProcCpuSecondsRef = useRef(null);
  const lastProcCpuSampleAtRef = useRef(null);
  const lastPredictedTotalRef = useRef(null);
  const eventSourceRef = useRef(null);

  const getActiveModelIdForMetrics = () => {
    // New client-only config (SettingsPage)
    try {
      const clientCfg = JSON.parse(localStorage.getItem('llmServerClientConfig')) || null;
      if (clientCfg && clientCfg.models && clientCfg.activeModelId && clientCfg.models[clientCfg.activeModelId]) {
        const modelPath = String(clientCfg.models[clientCfg.activeModelId].modelPath || '').trim();
        if (modelPath) return modelPath;
      }
    } catch (_e) {}

    // Legacy single-model config
    try {
      const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
      const modelPath = String(config.modelPath || '').trim();
      if (modelPath) return modelPath;
    } catch (_e) {}

    return '';
  };

  useEffect(() => {
    // 토큰 속도 업데이트를 위한 전역 이벤트 리스너
    const handleTokenReceived = () => {
      const now = Date.now();
      const timeDiff = now - lastTokenTimeRef.current;
      
      tokenCountRef.current += 1;
      
      // 1초마다 속도 계산
      if (timeDiff >= 1000) {
        const tokensPerSecond = (tokenCountRef.current / timeDiff) * 1000;
        tokenSpeedRef.current = tokensPerSecond;
        setTokenSpeed(tokensPerSecond);
        setTokenCount(tokenCountRef.current);
        tokenCountRef.current = 0;
        lastTokenTimeRef.current = now;
      }
    };

    // 전역 이벤트 리스너 등록
    window.addEventListener('token-received', handleTokenReceived);

    // Context 사용량 업데이트 리스너
    const handleContextUpdate = (event) => {
      const { used, total } = event.detail;
      setContextUsed(used);
      contextSizeRef.current = total;
      setContextSize(total);
      const usagePercent = total > 0 ? (used / total) * 100 : 0;
      setContextUsage(usagePercent);
    };

    window.addEventListener('context-update', handleContextUpdate);

    // 설정 변경 이벤트 리스너
    const handleConfigUpdate = (event) => {
      const { contextSize: newSize } = event.detail || {};
      if (newSize && newSize !== contextSizeRef.current) {
        contextSizeRef.current = newSize;
        setContextSize(newSize);
        // 현재 사용량 비율 유지하면서 최대값만 업데이트
        const currentPercent = contextUsage;
        const newUsed = Math.round((currentPercent / 100) * newSize);
        setContextUsed(newUsed);
      }
    };

    window.addEventListener('config-updated', handleConfigUpdate);

    // 초기 context size 로드
    const loadContextSize = () => {
      try {
        const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
        const size = config.contextSize || 2048;
        contextSizeRef.current = size;
        setContextSize(size);
      } catch (error) {
        console.error('Failed to load context size:', error);
      }
    };
    loadContextSize();

    // 시스템 리소스 모니터링
    const updateSystemMetrics = async () => {
      if (window.electronAPI && window.electronAPI.getSystemMetrics) {
        try {
          const metrics = await window.electronAPI.getSystemMetrics();
          setCpuUsage(metrics.cpu || 0);
          setMemoryUsage(metrics.memory || 0);
          
          // GPU 사용량: 실제 GPU 처리량 (기존 방식 유지)
          setGpuUsage(metrics.gpu || 0);
          
          // VRAM 사용량: 별도로 설정
          // console.log('[PerformancePanel] Received metrics:', {
          //   vramTotal: metrics.vramTotal,
          //   vramUsed: metrics.vramUsed,
          //   vramUsage: metrics.vramUsage,
          // });
          
          if (metrics.vramTotal) {
            setVramTotal(metrics.vramTotal);
            // console.log('[PerformancePanel] Set vramTotal:', metrics.vramTotal);
          }
          if (metrics.vramUsed !== undefined) {
            setVramUsed(metrics.vramUsed);
            // console.log('[PerformancePanel] Set vramUsed:', metrics.vramUsed);
          }
        } catch (error) {
          console.error('Failed to get system metrics:', error);
          // Fallback to mock data
          setCpuUsage(Math.random() * 100);
          setGpuUsage(Math.random() * 100);
          setMemoryUsage(Math.random() * 100);
        }
      } else {
        // client-only: use SSE stream from llama-server (no polling)
        // NOTE: this is a no-op here; SSE is established in a separate effect below.
      }
    };

    // localStorage 변경 감지를 위한 주기적 체크
    const checkConfigUpdate = () => {
      try {
        const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
        const size = config.contextSize || 2048;
        if (size !== contextSizeRef.current) {
          contextSizeRef.current = size;
          setContextSize(size);
          // 현재 사용량 비율 유지하면서 최대값만 업데이트
          const currentPercent = contextUsage;
          const newUsed = Math.round((currentPercent / 100) * size);
          setContextUsed(newUsed);
        }
      } catch (error) {
        console.error('Failed to check config update:', error);
      }
    };

    // Electron mode keeps polling via IPC. Client-only uses SSE below.
    const interval = window.electronAPI && window.electronAPI.getSystemMetrics
      ? setInterval(updateSystemMetrics, 1000)
      : null;
    const configCheckInterval = setInterval(checkConfigUpdate, 1000);

    updateSystemMetrics(); // Initial call (Electron only)

    return () => {
      window.removeEventListener('token-received', handleTokenReceived);
      window.removeEventListener('context-update', handleContextUpdate);
      window.removeEventListener('config-updated', handleConfigUpdate);
      if (interval) clearInterval(interval);
      clearInterval(configCheckInterval);
    };
  }, []);

  // SSE metrics stream (client-only mode)
  useEffect(() => {
    if (window.electronAPI && window.electronAPI.getSystemMetrics) return;
    if (typeof EventSource === 'undefined') return;

    const connect = () => {
      const modelId = getActiveModelIdForMetrics();
      if (!modelId) return;

      const url = `${LLAMA_BASE_URL}/metrics/stream?model=${encodeURIComponent(modelId)}&interval_ms=1000`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.addEventListener('metrics', (evt) => {
        try {
          const data = JSON.parse(evt.data);

          const vramTotalBytes = Number(data.vramTotal || 0);
          const vramUsedBytes = Number(data.vramUsed || 0);
          if (vramTotalBytes > 0) setVramTotal(Math.round(vramTotalBytes));
          if (vramUsedBytes >= 0) setVramUsed(Math.round(vramUsedBytes));

          const sysMemTotal = Number(data.sysMemTotal || 0);
          const sysMemUsed = Number(data.sysMemUsed || 0);
          if (sysMemTotal > 0 && sysMemUsed >= 0) {
            setMemoryUsage(Math.max(0, Math.min(100, (sysMemUsed / sysMemTotal) * 100)));
          }

          const now = Date.now();
          const cpuSec = Number(data.procCpuSec || 0);
          const cores = Math.max(1, Math.round(Number(data.cpuCores || 1)));
          if (lastProcCpuSecondsRef.current != null && lastProcCpuSampleAtRef.current != null) {
            const dt = (now - lastProcCpuSampleAtRef.current) / 1000;
            const dcpu = cpuSec - lastProcCpuSecondsRef.current;
            if (dt > 0 && dcpu >= 0) {
              const pct = (dcpu / dt / cores) * 100;
              setCpuUsage(Math.max(0, Math.min(100, pct)));
            }
          }
          lastProcCpuSecondsRef.current = cpuSec;
          lastProcCpuSampleAtRef.current = now;

          const tps = Number(data.tps || 0);
          setTokenSpeed(Math.max(0, tps));
          // keep existing "GPU" gauge semantics (display-only)
          setGpuUsage(Math.max(0, Math.min(100, Math.round(tps))));

          const predictedTotal = Number(data.predictedTotal || 0);
          if (lastPredictedTotalRef.current != null) {
            const delta = Math.max(0, Math.round(predictedTotal - lastPredictedTotalRef.current));
            setTokenCount(delta);
          }
          lastPredictedTotalRef.current = predictedTotal;
        } catch (_e) {
          // ignore
        }
      });
    };

    // connect immediately and reconnect when config changes
    connect();
    const onConfigUpdated = () => {
      if (eventSourceRef.current) {
        try { eventSourceRef.current.close(); } catch (_e) {}
        eventSourceRef.current = null;
      }
      connect();
    };

    window.addEventListener('client-config-updated', onConfigUpdated);
    window.addEventListener('config-updated', onConfigUpdated);
    window.addEventListener('storage', onConfigUpdated);

    return () => {
      window.removeEventListener('client-config-updated', onConfigUpdated);
      window.removeEventListener('config-updated', onConfigUpdated);
      window.removeEventListener('storage', onConfigUpdated);
      if (eventSourceRef.current) {
        try { eventSourceRef.current.close(); } catch (_e) {}
        eventSourceRef.current = null;
      }
    };
  }, []);

  // 반원형 게이지 컴포넌트
  const SemiCircleGauge = ({ value, label, maxValue = 100 }) => {
    const percentage = Math.min((value / maxValue) * 100, 100);
    const radius = 50;
    const centerX = 70;
    const centerY = 80;
    const startX = centerX - radius;
    const endX = centerX + radius;
    
    // 반원의 원주 길이 (π * radius)
    const circumference = Math.PI * radius;
    
    // 표시할 길이 계산 (비율에 따라)
    const offset = circumference - (percentage / 100) * circumference;
    
    // 고유한 그라데이션 ID 생성
    const gradientId = `gauge-gradient-${label.toLowerCase()}`;
    const useGradient = percentage > 50;

    return (
      <div className="gauge-container">
        <svg className="gauge-svg" viewBox="0 0 140 80">
          <defs>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgb(0, 100, 255)" />
              <stop offset="50%" stopColor="rgb(255, 255, 0)" />
              <stop offset="100%" stopColor="rgb(255, 0, 0)" />
            </linearGradient>
          </defs>
          {/* 배경 원호 */}
          <path
            d={`M ${startX} ${centerY} A ${radius} ${radius} 0 0 1 ${endX} ${centerY}`}
            fill="none"
            stroke="#333"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* 값 원호 */}
          <path
            d={`M ${startX} ${centerY} A ${radius} ${radius} 0 0 1 ${endX} ${centerY}`}
            fill="none"
            stroke={useGradient ? `url(#${gradientId})` : '#4CAF50'}
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
          />
        </svg>
        <div className="gauge-label">
          <div className="gauge-value">{Math.round(value)}%</div>
          <div className="gauge-name">{label}</div>
        </div>
      </div>
    );
  };

  // 토큰 속도 램프 (20개) - 그라데이션 색상
  const TokenSpeedLamps = ({ speed, maxSpeed = 100 }) => {
    const numLamps = 20;
    const activeLamps = Math.min(Math.round((speed / maxSpeed) * numLamps), numLamps);
    
    // 인덱스에 따라 색상 계산 (0: 파란색, 중간: 노란색, 끝: 빨간색)
    const getLampColor = (index, total) => {
      const ratio = index / (total - 1); // 0 ~ 1
      
      if (ratio <= 0.5) {
        // 파란색(0) -> 노란색(0.5)
        // 파란색: rgb(0, 100, 255), 노란색: rgb(255, 255, 0)
        const localRatio = ratio * 2; // 0 ~ 1
        const r = Math.round(localRatio * 255);
        const g = Math.round(100 + localRatio * 155); // 100 -> 255
        const b = Math.round(255 - localRatio * 255); // 255 -> 0
        return `rgb(${r}, ${g}, ${b})`;
      } else {
        // 노란색(0.5) -> 빨간색(1.0)
        // 노란색: rgb(255, 255, 0), 빨간색: rgb(255, 0, 0)
        const localRatio = (ratio - 0.5) * 2; // 0 ~ 1
        const r = 255;
        const g = Math.round(255 - localRatio * 255); // 255 -> 0
        const b = 0;
        return `rgb(${r}, ${g}, ${b})`;
      }
    };
    
    return (
      <div className="token-lamps-container">
        {Array.from({ length: numLamps }).map((_, index) => {
          const isActive = index < activeLamps;
          const color = getLampColor(index, numLamps);
          
          return (
            <div
              key={index}
              className={`token-lamp ${isActive ? 'active' : ''}`}
              style={isActive ? {
                backgroundColor: color,
                borderColor: color,
                boxShadow: `0 0 10px ${color}80`
              } : {}}
            />
          );
        })}
      </div>
    );
  };

  return (
    <div className="performance-panel">
      <div className="performance-grid">
        {/* CPU 게이지 */}
        <div className="performance-item">
          <SemiCircleGauge value={cpuUsage} label="CPU" />
        </div>

        {/* GPU 게이지 */}
        <div className="performance-item">
          <SemiCircleGauge value={gpuUsage} label="GPU" />
        </div>

        {/* 메모리 사용량 (System & GPU) */}
        <div className="performance-item memory-item">
          {/* System Memory */}
          <div className="memory-row">
            <div className="memory-label-small">System Memory</div>
            <div className="memory-bar-container small">
              <div 
                className="memory-bar" 
                style={{ 
                  width: `${memoryUsage}%`,
                  background: memoryUsage <= 50 
                    ? '#4CAF50' 
                    : 'linear-gradient(90deg, rgb(0, 100, 255), rgb(255, 255, 0), rgb(255, 0, 0))'
                }}
              />
              <span className="bar-value">{Math.round(memoryUsage)}%</span>
            </div>
          </div>
          
          {/* GPU Usage (VRAM 점유율) */}
          <div className="memory-row" style={{ marginTop: '5px' }}>
            <div className="memory-label-small">
              VRAM
              {vramTotal > 0 && (
                <span style={{ fontSize: '0.7em', display: 'block', fontWeight: 'normal' }}>
                  ({(vramUsed / 1024 / 1024 / 1024).toFixed(1)} / {(vramTotal / 1024 / 1024 / 1024).toFixed(1)} GB)
                </span>
              )}
            </div>
            <div className="memory-bar-container small">
              <div 
                className="memory-bar" 
                style={{ 
                  width: `${vramTotal > 0 && vramUsed !== undefined ? (vramUsed / vramTotal) * 100 : 0}%`,
                  background: (vramTotal > 0 && vramUsed !== undefined ? (vramUsed / vramTotal) * 100 : 0) <= 50 
                    ? '#4CAF50' 
                    : 'linear-gradient(90deg, rgb(0, 100, 255), rgb(255, 255, 0), rgb(255, 0, 0))'
                }}
              />
              <span className="bar-value">
                {vramTotal > 0 && vramUsed !== undefined ? Math.round((vramUsed / vramTotal) * 100) : 0}%
              </span>
            </div>
          </div>
        </div>

        {/* Context Windows 사용량 */}
        <div className="performance-item context-item">
          <div className="memory-label">Context Windows</div>
          <div className="memory-bar-container">
            <div 
              className="memory-bar" 
              style={{ 
                width: `${contextUsage}%`,
                background: contextUsage <= 50 
                  ? '#4CAF50' 
                  : 'linear-gradient(90deg, rgb(0, 100, 255), rgb(255, 255, 0), rgb(255, 0, 0))'
              }}
            />
            <span className="bar-value">{contextUsed} / {contextSize}</span>
          </div>
        </div>

        {/* 토큰 출력 속도 */}
        <div className="performance-item token-speed-item">
          <div className="token-speed-label">Token Speed</div>
          <div className="token-speed-value">{tokenSpeed.toFixed(1)} tokens/s</div>
          <TokenSpeedLamps speed={tokenSpeed} maxSpeed={50} />
        </div>

        {/* Token Debug Panel */}
        <div className="performance-item token-debug-item">
          <TokenDebugPanel />
        </div>
      </div>
    </div>
  );
};

export default PerformancePanel;
