import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import LanguageSelector from './LanguageSelector';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../contexts/AuthContext';
import { getActiveServerUrl, getActiveModelFormat } from '../services/api';
import './Header.css';

const Header = () => {
  const { t } = useLanguage();
  const auth = useAuth();
  const [config, setConfig] = useState({ models: [], activeModelId: null });
  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const STORAGE_KEY = 'llmServerClientConfig';
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isModelLoading, setIsModelLoading] = useState(false);

  // 페이지 로드 시 및 페이지 가시성 변경 시 서버의 현재 로딩 상태 확인
  useEffect(() => {
    const checkServerLoadingStatus = async (retryCount = 0) => {
      try {
        const serverUrl = getActiveServerUrl();
        const modelFormat = getActiveModelFormat();
        
        if (!serverUrl || !config?.activeModelId) return;
        
        // GGUF 서버의 경우 /loading-progress API 사용
        if (modelFormat === 'gguf') {
          try {
            const progressResponse = await fetch(`${serverUrl}/loading-progress`, {
              method: 'GET',
              signal: AbortSignal.timeout(2000),
            });
            if (progressResponse.ok) {
              const progressData = await progressResponse.json();
              if (progressData.loading) {
                setIsModelLoading(true);
                setLoadingProgress(progressData.progress || 0);
              } else {
                setIsModelLoading(false);
                setLoadingProgress(progressData.progress || 100);
              }
              return; // 프로그레스 정보를 받았으면 종료
            }
          } catch (_e) {
            // /loading-progress가 없으면 기존 방식 사용
          }
          
          // 기존 방식: /health 및 /models 확인
          try {
            const response = await fetch(`${serverUrl}/health`, {
              method: 'GET',
              signal: AbortSignal.timeout(2000),
            });
            
            // 503 상태 코드는 "Loading model"을 의미함
            if (response.status === 503) {
              setIsModelLoading(true);
              setLoadingProgress(0);
            } else if (response.ok) {
              // 서버가 준비되었으면 /models 엔드포인트에서 모델 상태 확인 (router 모드인 경우)
              try {
                const modelsResponse = await fetch(`${serverUrl}/models`, {
                  signal: AbortSignal.timeout(2000),
                });
                if (modelsResponse.ok) {
                  const modelsData = await modelsResponse.json();
                  const activeModelId = config?.activeModelId;
                  if (activeModelId && modelsData.data) {
                    const model = modelsData.data.find((m) => m.id === activeModelId);
                    if (model?.status?.value === 'loading') {
                      setIsModelLoading(true);
                      setLoadingProgress(0);
                    } else if (model?.status?.value === 'loaded') {
                      setIsModelLoading(false);
                      setLoadingProgress(100);
                    }
                  }
                } else {
                  // 단일 모델 모드인 경우, /health가 200이면 로딩 완료로 간주
                  setIsModelLoading(false);
                  setLoadingProgress(100);
                }
              } catch (_e) {
                // 단일 모델 모드인 경우, /health가 200이면 로딩 완료로 간주
                setIsModelLoading(false);
                setLoadingProgress(100);
              }
            }
          } catch (error) {
            // 연결 실패 시 재시도 (최대 5회, 1초 간격)
            if (retryCount < 5) {
              setTimeout(() => {
                checkServerLoadingStatus(retryCount + 1);
              }, 1000);
            }
          }
        } else {
          // MLX 서버의 경우
          try {
            const response = await fetch(`${serverUrl}/health`, {
              method: 'GET',
              signal: AbortSignal.timeout(2000),
            });
            
            // 503 상태 코드는 "Loading model"을 의미함
            if (response.status === 503) {
              setIsModelLoading(true);
              setLoadingProgress(0); // 초기 상태는 0%
            } else if (response.ok) {
              const data = await response.json();
              if (data.status === 'loading') {
                setIsModelLoading(true);
                // 서버에서 프로그레스 정보를 받으면 사용, 없으면 0%
                setLoadingProgress(data.progress !== undefined ? data.progress : 0);
              } else if (data.status === 'ready') {
                setIsModelLoading(false);
                setLoadingProgress(100); // 완료 상태
              }
            }
          } catch (error) {
            // 연결 실패 시 재시도 (최대 5회, 1초 간격)
            if (retryCount < 5) {
              setTimeout(() => {
                checkServerLoadingStatus(retryCount + 1);
              }, 1000);
            }
          }
        }
      } catch (_e) {
        // 서버가 아직 시작되지 않았거나 연결 불가 - 재시도
        if (retryCount < 5) {
          setTimeout(() => {
            checkServerLoadingStatus(retryCount + 1);
          }, 1000);
        }
      }
    };
    
    // config가 로드된 후 서버 상태 확인
    if (config?.activeModelId) {
      // 약간의 지연을 두어 서버가 시작될 시간을 줌
      const timer = setTimeout(() => {
        checkServerLoadingStatus();
      }, 1000);
      
      // 페이지 가시성 변경 시 상태 확인 (다른 앱에서 돌아올 때)
      const handleVisibilityChange = () => {
        if (!document.hidden && config?.activeModelId) {
          // 페이지가 다시 보일 때 서버 상태 확인 (서버가 시작될 시간을 줌)
          setTimeout(() => {
            checkServerLoadingStatus();
          }, 1000);
        }
      };
      
      document.addEventListener('visibilitychange', handleVisibilityChange);
      
      // 포커스 이벤트도 처리 (일부 브라우저에서 더 잘 동작)
      const handleFocus = () => {
        if (config?.activeModelId) {
          setTimeout(() => {
            checkServerLoadingStatus();
          }, 1000);
        }
      };
      
      window.addEventListener('focus', handleFocus);
      
      return () => {
        clearTimeout(timer);
        document.removeEventListener('visibilitychange', handleVisibilityChange);
        window.removeEventListener('focus', handleFocus);
      };
    }
  }, [config?.activeModelId]);

  useEffect(() => {
    let isInitialLoad = true;
    
    const loadConfig = async (maybeConfig) => {
      // console.log('[Header][DEBUG] ===== loadConfig called =====');
      // console.log('[Header][DEBUG] isInitialLoad:', isInitialLoad);
      // console.log('[Header][DEBUG] maybeConfig:', maybeConfig);
      // console.log('[Header][DEBUG] current config.activeModelId:', config?.activeModelId);
      
      let cfg = null;
      
      if (maybeConfig && maybeConfig.models) {
        // console.log('[Header][DEBUG] Using maybeConfig from parameter');
        // ensure activeModelId exists if models exist
        cfg = { ...maybeConfig };
        if (!cfg.activeModelId && Array.isArray(cfg.models) && cfg.models.length > 0) {
          // console.log('[Header][DEBUG] No activeModelId, setting to first model:', cfg.models[0].id);
          cfg.activeModelId = cfg.models[0].id;
        }
      } else if (window.electronAPI) {
        // console.log('[Header][DEBUG] Electron mode: Loading config via IPC');
        const loadedConfig = await window.electronAPI.loadConfig();
        // console.log('[Header][DEBUG] Electron mode: Loaded config:', loadedConfig);
        // Ensure loadedConfig and its models property are not null/undefined
        if (loadedConfig && loadedConfig.models) {
          cfg = loadedConfig;
          // ensure activeModelId exists if models exist
          if (!cfg.activeModelId && Array.isArray(cfg.models) && cfg.models.length > 0) {
            // console.log('[Header][DEBUG] Electron mode: No activeModelId, setting to first model:', cfg.models[0].id);
            cfg.activeModelId = cfg.models[0].id;
          }
        }
      } else {
        // console.log('[Header][DEBUG] Client mode: Loading config from localStorage');
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          // console.log('[Header][DEBUG] Client mode: Raw localStorage value:', raw ? 'exists' : 'null');
          if (raw) {
            const parsed = JSON.parse(raw);
            // console.log('[Header][DEBUG] Client mode: Parsed config:', parsed);
            if (parsed && parsed.models) {
              cfg = parsed;
              // ensure activeModelId exists if models exist
              if (!cfg.activeModelId && Array.isArray(cfg.models) && cfg.models.length > 0) {
                // console.log('[Header][DEBUG] Client mode: No activeModelId, setting to first model:', cfg.models[0].id);
                cfg.activeModelId = cfg.models[0].id;
              }
            }
          }
        } catch (_e) {
          // console.error('[Header][DEBUG] Client mode: Error parsing localStorage:', _e);
        }
      }
      
      // console.log('[Header][DEBUG] Final cfg:', cfg);
      // console.log('[Header][DEBUG] cfg.activeModelId:', cfg?.activeModelId);
      
      // config를 설정하고, activeModelId가 있으면 서버를 시작하도록 요청
      if (cfg && cfg.models) {
        const previousActiveModelId = config?.activeModelId;
        // console.log('[Header][DEBUG] Previous activeModelId:', previousActiveModelId);
        // console.log('[Header][DEBUG] New activeModelId:', cfg.activeModelId);
        // console.log('[Header][DEBUG] Models count:', cfg.models.length);
        
        setConfig(cfg);
        
        // activeModelId가 있으면 modelConfig도 업데이트하여 getActiveServerUrl()이 올바른 포트를 반환하도록 함
        if (cfg.activeModelId) {
          const activeModel = cfg.models.find(m => m.id === cfg.activeModelId);
          if (activeModel) {
            try {
              localStorage.setItem('modelConfig', JSON.stringify(activeModel));
              // config-updated 이벤트 발생시켜 PerformancePanel 등이 올바른 서버 URL로 재연결하도록 함
              window.dispatchEvent(new CustomEvent('config-updated', { 
                detail: { 
                  contextSize: activeModel.contextSize || 2048,
                  modelFormat: activeModel.modelFormat || 'gguf',
                  activeModelId: cfg.activeModelId
                } 
              }));
              // client-config-updated 이벤트도 발생시켜 다른 컴포넌트들이 config 변경을 감지하도록 함
              window.dispatchEvent(new CustomEvent('client-config-updated', { 
                detail: { config: cfg } 
              }));
            } catch (_e) {
              // ignore
            }
          }
        }
        
        // activeModelId가 있고, 이전과 다르거나 최초 로드인 경우 서버를 시작하도록 요청
        if (cfg.activeModelId) {
          const activeModel = cfg.models.find(m => m.id === cfg.activeModelId);
          // console.log('[Header][DEBUG] Active model found:', activeModel ? activeModel.id : 'NOT FOUND');
          
          if (activeModel) {
            // activeModelId가 변경되었거나 최초 로드인 경우 서버 시작
            const shouldStartServer = isInitialLoad || previousActiveModelId !== cfg.activeModelId;
            // console.log('[Header][DEBUG] shouldStartServer:', shouldStartServer);
            // console.log('[Header][DEBUG] - isInitialLoad:', isInitialLoad);
            // console.log('[Header][DEBUG] - previousActiveModelId !== cfg.activeModelId:', previousActiveModelId !== cfg.activeModelId);
            // console.log('[Header][DEBUG] - previousActiveModelId:', previousActiveModelId);
            // console.log('[Header][DEBUG] - cfg.activeModelId:', cfg.activeModelId);
            
            if (shouldStartServer) {
              // console.log('[Header][DEBUG] ✅ Starting server...');
              console.log('[Header] Config loaded/updated: Active model found, triggering server start:', {
                modelId: cfg.activeModelId,
                previousModelId: previousActiveModelId,
                isInitialLoad: isInitialLoad
              });
              
              // 초기 로드 시에는 서버가 이미 실행 중이므로 config만 저장
              if (window.electronAPI) {
                // console.log('[Header][DEBUG] Electron mode: Calling saveConfig...');
                try {
                  const result = await window.electronAPI.saveConfig(cfg);
                  // console.log('[Header][DEBUG] Electron mode: saveConfig result:', result);
                  console.log('[Header] Electron mode: Config saved');
                } catch (error) {
                  // console.error('[Header][DEBUG] Electron mode: saveConfig error:', error);
                  console.error('[Header] Electron mode: Failed to save config on load:', error);
                }
              } else {
                // 클라이언트 모드: /api/save-config 호출 (서버는 이미 실행 중)
                // console.log('[Header][DEBUG] Client mode: Calling /api/save-config...');
                try {
                  const response = await fetch('http://localhost:8083/api/save-config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(cfg),
                  });
                  if (response.ok) {
                    const result = await response.json();
                    // console.log('[Header][DEBUG] Client mode: Config saved:', result);
                  } else {
                    const errorText = await response.text();
                    // console.error('[Header][DEBUG] Client mode: Failed to save config:', response.status, errorText);
                  }
                } catch (error) {
                  // console.error('[Header][DEBUG] Client mode: Error saving config:', error);
                }
              }
            } else {
              // console.log('[Header][DEBUG] ❌ Skipping server start (shouldStartServer = false)');
              console.log('[Header] Config loaded but activeModelId unchanged, skipping server start');
            }
          } else {
            // console.error('[Header][DEBUG] ❌ Active model not found in models array');
            // console.error('[Header][DEBUG] Looking for modelId:', cfg.activeModelId);
            // console.error('[Header][DEBUG] Available model IDs:', cfg.models.map(m => m.id));
          }
        } else {
          // console.log('[Header][DEBUG] ❌ No activeModelId in config');
        }
      } else {
        // console.log('[Header][DEBUG] ❌ No config or models found');
        // console.log('[Header][DEBUG] cfg:', cfg);
        // console.log('[Header][DEBUG] cfg.models:', cfg?.models);
      }
      
      isInitialLoad = false;
      // console.log('[Header][DEBUG] ===== loadConfig completed =====');
    };
    loadConfig();

    const handleClientConfigUpdated = (event) => {
      // console.log('[Header][DEBUG] ===== handleClientConfigUpdated called =====');
      // console.log('[Header][DEBUG] event:', event);
      // console.log('[Header][DEBUG] event.detail:', event?.detail);
      const next = event?.detail?.config;
      // console.log('[Header][DEBUG] next config:', next);
      if (next) {
        // console.log('[Header][DEBUG] Calling loadConfig with next config');
        loadConfig(next);
      } else {
        // console.log('[Header][DEBUG] No next config, skipping loadConfig');
      }
    };

    const handleStorage = (event) => {
      // console.log('[Header][DEBUG] ===== handleStorage called =====');
      // console.log('[Header][DEBUG] event.key:', event?.key);
      // console.log('[Header][DEBUG] STORAGE_KEY:', STORAGE_KEY);
      if (event && event.key === STORAGE_KEY) {
        // console.log('[Header][DEBUG] Storage key matches, parsing newValue');
        try {
          if (event.newValue) {
            // console.log('[Header][DEBUG] event.newValue exists, length:', event.newValue.length);
            const parsed = JSON.parse(event.newValue);
            // console.log('[Header][DEBUG] Parsed storage value:', parsed);
            if (parsed && parsed.models) {
              // console.log('[Header][DEBUG] Parsed config has models, calling loadConfig');
              loadConfig(parsed);
            } else {
              // console.log('[Header][DEBUG] Parsed config missing models, skipping loadConfig');
            }
          } else {
            // console.log('[Header][DEBUG] event.newValue is null/undefined');
          }
        } catch (_e) {
          // console.error('[Header][DEBUG] Error parsing storage value:', _e);
        }
      } else {
        // console.log('[Header][DEBUG] Storage key does not match, ignoring');
      }
    };

    // 모델 로딩 상태 및 프로그레스 추적
    const handleModelLoading = (event) => {
      // 프로그레스 바를 항상 표시하도록 유지
      // setIsModelLoading(event.detail.loading);
      // if (!event.detail.loading) {
      //   setLoadingProgress(0);
      // }
    };

    const handleServerLog = (event) => {
      // progress 필드가 직접 전달된 경우 (MLX 서버의 type: "progress" 메시지)
      if (event.detail && typeof event.detail === 'object' && 'progress' in event.detail) {
        const progress = event.detail.progress;
        if (typeof progress === 'number') {
          setLoadingProgress(Math.min(100, Math.max(0, progress)));
          setIsModelLoading(true);
        }
        // text 필드도 확인하여 로딩 완료 감지
        const text = event.detail.text || '';
        if (text.includes('Model loaded successfully') || text.includes('✅ Model loaded')) {
          setLoadingProgress(100);
          setIsModelLoading(false);
        }
        return;
      }
      
      const logMessage = typeof event.detail === 'string' ? event.detail : '';
      if (!logMessage) return;
      
      // 로그에서 프로그레스 정보 파싱: "Loading progress: [████████░░░░░░░░░░░░░░░░░░░░] 45.2%"
      const progressMatch = logMessage.match(/Loading progress:.*?(\d+\.?\d*)%/);
      if (progressMatch) {
        const progress = parseFloat(progressMatch[1]);
        setLoadingProgress(Math.min(100, Math.max(0, progress)));
        setIsModelLoading(true);
      }
      // 모델 로딩 완료 감지 - 프로그레스 바는 유지하되 100%로 설정
      if (logMessage.includes('Model loaded successfully') || 
          logMessage.includes('✅ Model loaded') ||
          logMessage.includes('model loaded') ||
          /model\s+loaded/i.test(logMessage)) {
        setLoadingProgress(100);
        setIsModelLoading(false); // 로딩 완료
      }
      // 모델 로딩 시작 감지 (GGUF: "loading model", MLX: "Loading model from" 등)
      if (logMessage.includes('Loading model from') || 
          logMessage.includes('Loading...') ||
          /loading\s+model/i.test(logMessage)) {
        setIsModelLoading(true);
        if (!progressMatch) {
          setLoadingProgress(0);
        }
      }
    };

    window.addEventListener('client-config-updated', handleClientConfigUpdated);
    window.addEventListener('storage', handleStorage);
    window.addEventListener('model-loading', handleModelLoading);
    window.addEventListener('server-log', handleServerLog);
    
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      window.removeEventListener('client-config-updated', handleClientConfigUpdated);
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener('model-loading', handleModelLoading);
      window.removeEventListener('server-log', handleServerLog);
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  const handleSelectModel = async (modelId) => {
    console.log('[Header] ===== Model Selection Started =====');
    console.log('[Header] Selected model ID:', modelId);
    
    const newConfig = { ...config, activeModelId: modelId };
    const activeModel = newConfig.models?.find(m => m.id === modelId);
    
    // 바로 모델 변경 수행 (서버는 자동으로 적절한 포트로 요청됨)
    await performModelSwitch(modelId, newConfig, activeModel);
  };

  const performModelSwitch = async (modelId, newConfig, activeModel) => {
    // config를 항상 업데이트하여 드롭다운이 올바른 모델을 표시하도록 함
    console.log('[Header] performModelSwitch: Updating config', {
      modelId,
      newActiveModelId: newConfig.activeModelId,
      currentActiveModelId: config.activeModelId
    });
    setConfig(newConfig);
    setDropdownOpen(false);
    
    if (window.electronAPI) {
      console.log('[Header] Electron mode: Saving config via IPC...');
      try {
        const result = await window.electronAPI.saveConfig(newConfig);
        console.log('[Header] Electron mode: Config save result:', result);
        if (result.success) {
          console.log('[Header] ✅ Config saved successfully');
        } else {
          console.error('[Header] ❌ Config save failed:', result.error);
        }
      } catch (error) {
        console.error('[Header] ❌ Error saving config via IPC:', error);
      }
    } else {
      // 클라이언트 모드: config.json을 업데이트하여 서버 전환 트리거
      console.log('[Header] Client mode: Saving config to localStorage and API...');
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));
        console.log('[Header] Client mode: localStorage updated');
        
        // start-client-server.js의 /api/save-config 엔드포인트 호출
        console.log('[Header] Client mode: Calling /api/save-config...');
        const response = await fetch('http://localhost:8083/api/save-config', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(newConfig),
        });
        
        if (response.ok) {
          const result = await response.json();
          console.log('[Header] ✅ Client mode: Config saved to server:', result);
        } else {
          const errorText = await response.text();
          console.error('[Header] ❌ Client mode: Failed to save config:', response.status, errorText);
        }
      } catch (error) {
        console.error('[Header] ❌ Client mode: Error saving config:', error);
      }
    }
    
    console.log('[Header] ===== Model Selection Completed =====');

    // 클라이언트 추론 파라미터용 active model도 갱신
    if (activeModel) {
      try {
        localStorage.setItem('modelConfig', JSON.stringify(activeModel));
        const contextSize = activeModel.contextSize || 2048;
        const modelFormat = activeModel.modelFormat || 'gguf';
        // config-updated 이벤트 발생시켜 PerformancePanel 등이 올바른 서버 URL로 재연결하도록 함
        window.dispatchEvent(new CustomEvent('config-updated', { 
          detail: { 
            contextSize,
            modelFormat,
            activeModelId: modelId
          } 
        }));
        // client-config-updated 이벤트도 발생시켜 다른 컴포넌트들이 config 변경을 감지하도록 함
        window.dispatchEvent(new CustomEvent('client-config-updated', { 
          detail: { config: newConfig } 
        }));
      } catch (_e) {
        // ignore
      }
    }
  };


  // Defensive coding: ensure config and config.models exist before trying to use them
  const models = config?.models || [];
  const activeModel = models.find(m => m.id === config?.activeModelId);
  const getModelLabel = (m) => (m?.modelPath || m?.name || '').trim();


  return (
    <>
      <header className="app-header">
        <div className="header-left">
          <h1>LLM Lab</h1>
          <nav>
            <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
              Chat
            </NavLink>
            <NavLink to="/guide" className={({ isActive }) => (isActive ? 'active' : '')}>
              {t('header.guide')}
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}>
              {t('header.settings')}
            </NavLink>
          </nav>
          <div className="model-selector-dropdown" ref={dropdownRef}>
            <button className="current-model-display" onClick={() => setDropdownOpen(!isDropdownOpen)}>
              {activeModel ? (getModelLabel(activeModel) || t('header.selectModel')) : t('header.selectModel')}
              <span className="dropdown-arrow">{isDropdownOpen ? '▲' : '▼'}</span>
            </button>
            {isDropdownOpen && (
              <div className="model-dropdown-content">
                {models.map(model => (
                  <div 
                    key={model.id} 
                    className="model-dropdown-item"
                    onClick={() => handleSelectModel(model.id)}
                  >
                    {getModelLabel(model) || model.id}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="model-loading-progress-container">
            <div className="model-loading-progress-bar">
              <div 
                className="model-loading-progress-fill"
                style={{ width: `${loadingProgress}%` }}
              />
            </div>
          </div>
        </div>

        <div className="header-center">
        </div>

        <div className="header-right">
          <LanguageSelector />
          {auth?.authenticated ? (
            <button
              className="logout-icon-button"
              title={t('header.logout')}
              onClick={() => auth.logout()}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="currentColor"
                  d="M10 17v-2h4v-6h-4V7l-5 5 5 5zm9-14h-8v2h8v14h-8v2h8a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"
                />
              </svg>
            </button>
          ) : null}
        </div>
      </header>
    </>
  );
};

export default Header;
