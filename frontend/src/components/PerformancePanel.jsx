import React, { useState, useEffect, useRef } from 'react';
import './PerformancePanel.css';
import TokenDebugPanel from './TokenDebugPanel';

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
          if (metrics.vramTotal) setVramTotal(metrics.vramTotal);
          if (metrics.vramUsed !== undefined) setVramUsed(metrics.vramUsed);
        } catch (error) {
          console.error('Failed to get system metrics:', error);
          // Fallback to mock data
          setCpuUsage(Math.random() * 100);
          setGpuUsage(Math.random() * 100);
          setMemoryUsage(Math.random() * 100);
        }
      } else {
        // Fallback to mock data if Electron API not available
        setCpuUsage(Math.random() * 100);
        setGpuUsage(Math.random() * 100);
        setMemoryUsage(Math.random() * 100);
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

    const interval = setInterval(updateSystemMetrics, 1000);
    const configCheckInterval = setInterval(checkConfigUpdate, 1000);

    updateSystemMetrics(); // Initial call

    return () => {
      window.removeEventListener('token-received', handleTokenReceived);
      window.removeEventListener('context-update', handleContextUpdate);
      window.removeEventListener('config-updated', handleConfigUpdate);
      clearInterval(interval);
      clearInterval(configCheckInterval);
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
