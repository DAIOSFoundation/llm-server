import React, { useState, useEffect, useRef } from 'react';
import './PerformancePanel.css';
import TokenDebugPanel from './TokenDebugPanel';
import { getActiveServerUrl, getActiveModelFormat } from '../services/api';

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
  const lastTokenTimeRef = useRef(0); // 첫 토큰이 올 때 초기화
  const tokenCountRef = useRef(0);
  const lastUpdateTimeRef = useRef(0); // 마지막 업데이트 시간
  const contextSizeRef = useRef(2048);
  const lastProcCpuSecondsRef = useRef(null);
  const lastProcCpuSampleAtRef = useRef(null);
  const lastPredictedTotalRef = useRef(null);
  const eventSourceRef = useRef(null);
  const websocketRef = useRef(null);

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
      
      // 첫 토큰인 경우 시간 초기화
      if (lastTokenTimeRef.current === 0) {
        lastTokenTimeRef.current = now;
        tokenCountRef.current = 1; // 첫 토큰도 카운트에 포함
        lastUpdateTimeRef.current = now;
        // console.log('[PerformancePanel] First token received, initializing timer');
        return; // 첫 토큰은 시간만 설정하고 계산하지 않음
      }
      
      tokenCountRef.current += 1;
      const timeDiff = now - lastTokenTimeRef.current;
      const timeSinceLastUpdate = now - lastUpdateTimeRef.current;
      
      // 30ms 이상 경과했거나, 토큰이 10개 이상 모이고 마지막 업데이트 이후 200ms가 지났을 때 속도 계산
      if (timeDiff >= 30 || (tokenCountRef.current >= 10 && timeSinceLastUpdate >= 200)) {
        const tokensPerSecond = (tokenCountRef.current / timeDiff) * 1000;
        if (tokensPerSecond > 0) {
          tokenSpeedRef.current = tokensPerSecond;
          setTokenSpeed(tokensPerSecond);
          setTokenCount(tokenCountRef.current);
          lastUpdateTimeRef.current = now;
          // 디버깅용
          // console.log('[PerformancePanel] Token speed calculated:', tokensPerSecond.toFixed(2), 'tokens:', tokenCountRef.current, 'timeDiff:', timeDiff, 'ms');
        }
        tokenCountRef.current = 0;
        lastTokenTimeRef.current = now;
      }
    };

    // 전역 이벤트 리스너 등록
    window.addEventListener('token-received', handleTokenReceived);
    // console.log('[PerformancePanel] token-received event listener registered');

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
      // 리셋
      lastTokenTimeRef.current = 0;
      tokenCountRef.current = 0;
      lastUpdateTimeRef.current = 0;
    };
  }, []);

  // Metrics stream (SSE for GGUF, WebSocket for MLX)
  useEffect(() => {
    if (window.electronAPI && window.electronAPI.getSystemMetrics) return;
    
    const modelFormat = getActiveModelFormat();
    const useWebSocket = modelFormat === 'mlx';
    
    if (useWebSocket && typeof WebSocket === 'undefined') return;
    if (!useWebSocket && typeof EventSource === 'undefined') return;

    let stopped = false;
    let reconnectTimer = null;
    let probeTimer = null;
    let currentModelId = '';
    let currentServerUrl = '';

    const close = () => {
      if (eventSourceRef.current) {
        try { eventSourceRef.current.close(); } catch (_e) {}
        eventSourceRef.current = null;
      }
      if (websocketRef.current) {
        try { websocketRef.current.close(); } catch (_e) {}
        websocketRef.current = null;
      }
    };

    const scheduleReconnect = (delayMs = 3000) => {
      if (stopped) return;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        if (!stopped) {
          connect();
        }
      }, delayMs);
    };

    const connect = () => {
      if (stopped) return;

      const modelId = getActiveModelIdForMetrics();
      if (!modelId) {
        currentModelId = '';
        currentServerUrl = '';
        close();
        return;
      }

      // 현재 서버 URL 가져오기
      const serverUrl = getActiveServerUrl();

      // 이미 같은 모델과 서버 URL로 연결되어 있으면 재연결하지 않음
      if (useWebSocket) {
        if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN && 
            currentModelId === modelId && currentServerUrl === serverUrl) {
          // console.log('[PerformancePanel] WebSocket already connected, skipping');
          return;
        }
      } else {
        if (eventSourceRef.current && currentModelId === modelId && currentServerUrl === serverUrl) {
          return;
        }
      }

      // 모델이나 서버 URL이 변경되었으면 재연결
      currentModelId = modelId;
      currentServerUrl = serverUrl;
      close();

      const handleMetricsData = (data) => {
        try {
          // console.log('[PerformancePanel] handleMetricsData called with:', data);
          
          const vramTotalBytes = Number(data.vramTotal || 0);
          const vramUsedBytes = Number(data.vramUsed || 0);
          // console.log('[PerformancePanel] VRAM:', { vramTotalBytes, vramUsedBytes });
          if (vramTotalBytes > 0) {
            setVramTotal(Math.round(vramTotalBytes));
            // console.log('[PerformancePanel] Set vramTotal:', Math.round(vramTotalBytes));
          }
          if (vramUsedBytes >= 0) {
            setVramUsed(Math.round(vramUsedBytes));
            // console.log('[PerformancePanel] Set vramUsed:', Math.round(vramUsedBytes));
          }

          const sysMemTotal = Number(data.sysMemTotal || 0);
          const sysMemUsed = Number(data.sysMemUsed || 0);
          // console.log('[PerformancePanel] System Memory:', { sysMemTotal, sysMemUsed });
          if (sysMemTotal > 0 && sysMemUsed >= 0) {
            const memUsage = Math.max(0, Math.min(100, (sysMemUsed / sysMemTotal) * 100));
            setMemoryUsage(memUsage);
            // console.log('[PerformancePanel] Set memoryUsage:', memUsage);
          }

          const now = Date.now();
          const cpuSec = Number(data.procCpuSec || 0);
          const cores = Math.max(1, Math.round(Number(data.cpuCores || 1)));
          // console.log('[PerformancePanel] CPU:', { cpuSec, cores, lastProcCpuSecondsRef: lastProcCpuSecondsRef.current });
          if (lastProcCpuSecondsRef.current != null && lastProcCpuSampleAtRef.current != null) {
            const dt = (now - lastProcCpuSampleAtRef.current) / 1000;
            const dcpu = cpuSec - lastProcCpuSecondsRef.current;
            if (dt > 0 && dcpu >= 0) {
              const pct = (dcpu / dt / cores) * 100;
              setCpuUsage(Math.max(0, Math.min(100, pct)));
              // console.log('[PerformancePanel] Set cpuUsage:', pct);
            }
          } else {
            // 첫 번째 메트릭이므로 초기값 설정만
            // console.log('[PerformancePanel] First metrics, setting initial values');
          }
          lastProcCpuSecondsRef.current = cpuSec;
          lastProcCpuSampleAtRef.current = now;

          const tps = Number(data.tps || 0);
          // 클라이언트 측 계산 값이 있으면 우선 사용 (더 정확한 실시간 속도)
          // 클라이언트 측 계산이 더 최근 데이터를 반영하므로 항상 우선순위가 높음
          if (tokenSpeedRef.current > 0) {
            setTokenSpeed(tokenSpeedRef.current);
          } else if (tps > 0) {
            // 클라이언트 측 계산이 없으면 서버 메트릭 사용
            setTokenSpeed(Math.max(0, tps));
            tokenSpeedRef.current = tps;
            // console.log('[PerformancePanel] Using server tps:', tps);
          }
          // 디버깅용
          if (tps > 0 || tokenSpeedRef.current > 0) {
            // console.log('[PerformancePanel] Token speed - server:', tps, 'client:', tokenSpeedRef.current, 'final:', tokenSpeedRef.current > 0 ? tokenSpeedRef.current : tps);
          }
          
          // GPU 게이지는 실제 GPU 점유율을 얻기 어려워(플랫폼별/백엔드별),
          // router 모드에서는 VRAM 점유율(%)을 GPU 지표로 사용한다.
          if (vramTotalBytes > 0) {
            const gpuUsage = Math.max(0, Math.min(100, (vramUsedBytes / vramTotalBytes) * 100));
            setGpuUsage(gpuUsage);
            // console.log('[PerformancePanel] Set gpuUsage:', gpuUsage);
          } else {
            setGpuUsage(0);
            // console.log('[PerformancePanel] Set gpuUsage: 0 (no VRAM total)');
          }

          const predictedTotal = Number(data.predictedTotal || 0);
          if (lastPredictedTotalRef.current != null) {
            const delta = Math.max(0, Math.round(predictedTotal - lastPredictedTotalRef.current));
            setTokenCount(delta);
            // console.log('[PerformancePanel] Set tokenCount:', delta);
          }
          lastPredictedTotalRef.current = predictedTotal;
        } catch (e) {
          // console.error('[PerformancePanel] Error in handleMetricsData:', e, data);
        }
      };

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
                const wsUrlFull = `${wsUrl}/metrics/stream`;
                const ws = new WebSocket(wsUrlFull);
                websocketRef.current = ws;

                ws.onopen = () => {
                  // WebSocket 연결 성공
                };

                ws.onmessage = (event) => {
                  try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics') {
                      handleMetricsData(data);
                    }
                  } catch (e) {
                    // ignore parse errors
                  }
                };

                ws.onerror = () => {
                  // 에러는 조용히 처리, onclose에서 재연결
                };

                ws.onclose = (event) => {
                  if (!stopped && event.code !== 1000) {  // 정상 종료(1000)가 아닌 경우만 재연결
                    scheduleReconnect(5000);  // 재연결 간격 증가
                  }
                };
              } else {
                // 서버가 준비되지 않았으면 재시도
                scheduleReconnect(5000);
              }
            } else {
              // 헬스 체크 실패 시 재시도
              scheduleReconnect(5000);
            }
          } catch (e) {
            // 서버가 아직 시작되지 않았거나 연결 불가 - 재시도
            scheduleReconnect(5000);
          }
        };
        
        checkAndConnect();
      } else {
        // GGUF 모델: SSE 사용
        // autoload=1 ensures router spawns the model process even if not already loaded
        const url = `${serverUrl}/metrics/stream?model=${encodeURIComponent(modelId)}&interval_ms=1000&autoload=1`;
        const es = new EventSource(url);
        eventSourceRef.current = es;

        es.addEventListener('metrics', (evt) => {
          try {
            const data = JSON.parse(evt.data);
            handleMetricsData(data);
          } catch (_e) {
            // ignore
          }
        });

        // if the SSE connection drops (e.g. router/model restart), retry automatically
        es.onerror = () => {
          close();
          scheduleReconnect(1000);
        };
      }
    };

    // connect immediately and reconnect when config changes
    connect();
    const onConfigUpdated = () => {
      // config가 변경되었으면 현재 연결 정보를 초기화하고 재연결
      currentModelId = '';
      currentServerUrl = '';
      close();
      // 약간의 지연을 두어 localStorage가 완전히 업데이트되도록 함
      setTimeout(() => {
        connect();
      }, 100);
    };

    window.addEventListener('client-config-updated', onConfigUpdated);
    window.addEventListener('config-updated', onConfigUpdated);
    window.addEventListener('storage', onConfigUpdated);

    // Fallback: when localStorage is updated in the same tab, the 'storage' event doesn't fire.
    // Probe the active model id and server URL locally and connect once they change.
    probeTimer = window.setInterval(() => {
      if (stopped) return;
      const modelId = getActiveModelIdForMetrics();
      const serverUrl = getActiveServerUrl();
      if (modelId && (!eventSourceRef.current || modelId !== currentModelId || serverUrl !== currentServerUrl)) {
        connect();
      }
    }, 1000);

    return () => {
      stopped = true;
      window.removeEventListener('client-config-updated', onConfigUpdated);
      window.removeEventListener('config-updated', onConfigUpdated);
      window.removeEventListener('storage', onConfigUpdated);
      if (probeTimer) window.clearInterval(probeTimer);
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      close();
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
          <div className="gauge-value">
            {(() => {
              const v = Number.isFinite(value) ? value : 0;
              // 작은 값(유휴 상태에서도 발생하는 미세 사용률)이 0%로 보이지 않도록 소수점 표기
              const digits = v < 1 ? 2 : v < 10 ? 1 : 0;
              return `${v.toFixed(digits)}%`;
            })()}
          </div>
          <div className="gauge-name">{label}</div>
        </div>
      </div>
    );
  };

  // 토큰 속도 램프 (15개) - 그라데이션 색상
  const TokenSpeedLamps = ({ speed, maxSpeed = 100 }) => {
    const numLamps = 15;
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
