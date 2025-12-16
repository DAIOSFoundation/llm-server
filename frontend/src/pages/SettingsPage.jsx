import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import ModelForm from '../components/ModelForm';
import { useLanguage } from '../contexts/LanguageContext';
import './SettingsPage.css';
import { LLAMA_BASE_URL } from '../services/api';

const DescriptionCard = ({ title, content }) => (
  <div className="description-card">
    <h4>{title}</h4>
    <p>{content}</p>
  </div>
);

const SettingsPage = () => {
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [config, setConfig] = useState({ models: [], activeModelId: null });
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [isDescriptionsExpanded, setIsDescriptionsExpanded] = useState(false);

  const STORAGE_KEY = 'llmServerClientConfig';

  const deriveModelIdFromAnyPath = (p) => {
    if (!p) return '';
    const file = String(p).split(/[\\/]/).pop() || '';
    return file.replace(/\.gguf$/i, '');
  };

  const migrateLegacyModelsToClientConfig = (legacyModels) => {
    const migrated = (legacyModels || []).map((m, idx) => {
      const ic = m?.inferenceConfig || {};
      const modelId = deriveModelIdFromAnyPath(m?.path) || '';
      const legacyDevice = String(m?.device || '').toUpperCase();
      const legacyAccel = (legacyDevice === 'MPS') ? 'mps' : 'auto';

      return {
        id: m?.id || `legacy_${Date.now()}_${idx}`,
        // name은 UI에서 거의 쓰지 않지만 보존
        name: m?.name || 'Legacy Model',
        // 라우터 모드에서 쓰는 "모델 ID"
        modelPath: modelId,
        accelerator: legacyAccel,
        // MPS(=Metal)로 쓰던 레거시는 기본적으로 GPU offload를 켜는 것이 자연스럽습니다.
        // (-1 = auto/최대 offload)
        gpuLayers: legacyDevice === 'MPS' ? -1 : 0,
        contextSize: ic.contextSize ?? 2048,
        maxTokens: ic.maxTokens ?? 300,
        temperature: ic.temperature ?? 0.7,
        topK: ic.topK ?? 40,
        topP: ic.topP ?? 0.9,
        minP: 0.1,
        tfsZ: 1.0,
        typicalP: 1.0,
        repeatPenalty: ic.repeatPenalty ?? 1.2,
        repeatLastN: ic.repeatLastN ?? 128,
        penalizeNL: false,
        presencePenalty: 0.0,
        frequencyPenalty: 0.0,
        dryMultiplier: 0.8,
        dryBase: 1.75,
        dryAllowedLength: 2,
        dryPenaltyLastN: -1,
        mirostatMode: 0,
        mirostatTau: 5.0,
        mirostatEta: 0.1,
        showSpecialTokens: false,
      };
    }).filter((m) => m.modelPath);

    if (migrated.length === 0) return { models: [], activeModelId: null };
    return { models: migrated, activeModelId: migrated[0].id };
  };

  const loadClientConfig = () => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return { models: [], activeModelId: null };
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object' && Array.isArray(parsed.models)) {
        return parsed;
      }
    } catch (_e) {
      // ignore
    }
    return { models: [], activeModelId: null };
  };

  const loadClientConfigAsync = async () => {
    const local = loadClientConfig();
    if (local && Array.isArray(local.models) && local.models.length > 0) return local;

    // 레거시 파일(이전 버전)에서 1회 마이그레이션
    try {
      const res = await fetch('/legacy-models.json', { signal: AbortSignal.timeout(1000) });
      if (res.ok) {
        const legacy = await res.json();
        const migrated = migrateLegacyModelsToClientConfig(legacy);
        if (migrated.models.length > 0) {
          saveClientConfig(migrated);
          return migrated;
        }
      }
    } catch (_e) {
      // ignore
    }

    return local;
  };

  const saveClientConfig = (cfg) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg));
    } catch (_e) {
      // ignore
    }
  };

  useEffect(() => {
    const loadConfig = async () => {
      if (window.electronAPI) {
        const loadedConfig = await window.electronAPI.loadConfig();
          if (loadedConfig && loadedConfig.models) {
            // 강제 보정: 모델 설정 최적화 (반복 및 메타 발언 방지)
            const migratedModels = loadedConfig.models.map(model => {
              let changed = false;
              const newModel = { ...model };
              
              // 1. Repeat Penalty: 1.2 (기본 반복 억제 강화)
              if (newModel.repeatPenalty !== 1.2) {
                newModel.repeatPenalty = 1.2;
                changed = true;
              }
              // 2. Presence Penalty: 0.0 (초기화)
              if (newModel.presencePenalty !== 0.0) {
                newModel.presencePenalty = 0.0;
                changed = true;
              }
              // 3. Frequency Penalty: 0.0 (초기화)
              if (newModel.frequencyPenalty !== 0.0) {
                newModel.frequencyPenalty = 0.0;
                changed = true;
              }
              // 5. Min P 상향: 0.1 (잡담 제거)
              if (newModel.minP !== 0.1) {
                newModel.minP = 0.1;
                changed = true;
              }
              // 6. Top P 하향: 0.9 (확실한 단어만 선택)
              if (newModel.topP !== 0.9) {
                newModel.topP = 0.9;
                changed = true;
              }
              // 8. Penalize Newline: false (줄바꿈 허용 -> 자연스러운 종료 유도)
              if (newModel.penalizeNL !== false) {
                newModel.penalizeNL = false;
                changed = true;
              }
              if (newModel.dryBase !== 1.75) {
                newModel.dryBase = 1.75;
                changed = true;
              }
              if (newModel.dryAllowedLength !== 2) {
                newModel.dryAllowedLength = 2;
                changed = true;
              }
              // 5. Max Tokens: 300 (적당한 길이)
              if (newModel.maxTokens > 300) {
                newModel.maxTokens = 300;
                changed = true;
              }
              
              return newModel;
            });

            const newConfig = { ...loadedConfig, models: migratedModels };
            setConfig(newConfig);
            
            // 변경사항이 있으면 저장
            if (JSON.stringify(migratedModels) !== JSON.stringify(loadedConfig.models)) {
               window.electronAPI.saveConfig(newConfig);
            }

            if (loadedConfig.activeModelId) {
              setSelectedModelId(loadedConfig.activeModelId);
            } else if (loadedConfig.models.length > 0) {
              setSelectedModelId(loadedConfig.models[0].id);
            }
          }
      } else {
        const loadedConfig = await loadClientConfigAsync();
        setConfig(loadedConfig);
        if (loadedConfig.activeModelId) {
          setSelectedModelId(loadedConfig.activeModelId);
        } else if (loadedConfig.models.length > 0) {
          setSelectedModelId(loadedConfig.models[0].id);
        }
      }
    };
    loadConfig();
  }, []);

  const handleSelectModel = async (modelId) => {
    setSelectedModelId(modelId);
    // 드롭다운/저장 기준 active 모델도 함께 갱신
    const newConfig = { ...(config || { models: [], activeModelId: null }), activeModelId: modelId };
    setConfig(newConfig);
    
    // Electron 모드에서는 즉시 서버 전환
    if (window.electronAPI) {
      try {
        await window.electronAPI.saveConfig(newConfig);
      } catch (error) {
        console.error('Failed to save config:', error);
      }
    } else {
      // 클라이언트 모드에서는 localStorage에만 저장
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));
        window.dispatchEvent(new CustomEvent('client-config-updated', { detail: { config: newConfig } }));
      } catch (_e) {
        // ignore
      }
    }
  };
  
  const handleAddNewModel = () => {
    const newModel = {
      id: `model_${Date.now()}`,
      name: 'New Model',
      modelPath: '',
      modelFormat: 'gguf',
      accelerator: 'auto',
      gpuLayers: -1,
      contextSize: 2048,
      maxTokens: 600,
      temperature: 0.7,
      topK: 40,
      topP: 0.95,
      minP: 0.05,
      tfsZ: 1.0,
      typicalP: 1.0,
      repeatPenalty: 1.2,
      repeatLastN: 128,
      penalizeNL: true,
      presencePenalty: 0.3,
      frequencyPenalty: 0.5,
      dryMultiplier: 0.8,
      dryBase: 1.75,
      dryAllowedLength: 2,
      dryPenaltyLastN: -1,
      mirostatMode: 0,
      mirostatTau: 5.0,
      mirostatEta: 0.1,
      showSpecialTokens: false,
    };
    setConfig(prev => {
      const base = prev || { models: [], activeModelId: null };
      const nextModels = [...(base.models || []), newModel];
      return { ...base, models: nextModels, activeModelId: newModel.id };
    });
    setSelectedModelId(newModel.id);
  };
  
  const handleModelFormChange = (updatedModel) => {
    setConfig(prev => {
      const base = prev || { models: [], activeModelId: null };
      const newModels = (base.models || []).map(m => m.id === updatedModel.id ? updatedModel : m);
      return { ...base, models: newModels };
    });
  };
  
  const handleDeleteModel = (modelId) => {
    const newModels = (config.models || []).filter(m => m.id !== modelId);
    let newActiveModelId = config.activeModelId;
    if (config.activeModelId === modelId) {
      newActiveModelId = newModels.length > 0 ? newModels[0].id : null;
    }
    setConfig({ models: newModels, activeModelId: newActiveModelId });
    if (selectedModelId === modelId) {
      setSelectedModelId(newActiveModelId);
    }
  };

  const handleSave = async () => {
    let configToSave = { ...(config || { models: [], activeModelId: null }) };
    if (!configToSave.activeModelId && configToSave.models && configToSave.models.length > 0) {
      configToSave.activeModelId = configToSave.models[0].id;
    }

    if (window.electronAPI) {
      const result = await window.electronAPI.saveConfig(configToSave);
      if (!result.success) {
        alert(`Failed to save settings: ${result.error}`);
        return;
      }
    } else {
      saveClientConfig(configToSave);
      // 클라이언트 모드에서는 config.json 파일도 저장 (서버 시작용)
      try {
        const response = await fetch('http://localhost:8083/api/save-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(configToSave),
        }).catch(() => {});
        if (response && !response.ok) {
          console.warn('Failed to save config.json file for server');
        }
      } catch (error) {
        console.warn('Failed to save config.json file:', error);
      }
      // Header 드롭다운이 즉시 갱신되도록 이벤트로 알림
      window.dispatchEvent(new CustomEvent('client-config-updated', { detail: { config: configToSave } }));
    }

    // 설정 저장 후 localStorage에 저장 (Chat/Performance가 읽을 수 있도록)
    const activeModel = configToSave.models.find(m => m.id === configToSave.activeModelId);
    if (activeModel) {
      localStorage.setItem('modelConfig', JSON.stringify(activeModel));

      const contextSize = activeModel.contextSize || 2048;
      window.dispatchEvent(new CustomEvent('config-updated', {
        detail: { contextSize }
      }));
      window.dispatchEvent(new CustomEvent('config-updated'));
    }

    // 서버 재기동(라우터 모드): active model을 unload/load 해서 모델 로드 상태를 갱신
    // NOTE: 현재 llama-server 빌드에서는 extra_args(추가 실행 인자) 전달 기능을 사용하지 않습니다.
    if (!window.electronAPI && activeModel && activeModel.modelPath) {
      const model = String(activeModel.modelPath).trim();
      if (model) {
        try {
          const accel = String(activeModel.accelerator || 'auto').toLowerCase();
          const defaultGpuLayers = accel === 'cpu' ? 0 : -1;
          const gpuLayers =
            (activeModel.gpuLayers === '' || activeModel.gpuLayers == null)
              ? defaultGpuLayers
              : Number(activeModel.gpuLayers);

          // 서버에 모델별 로드 설정을 저장 (서버 측 JSON 파일에 기록됨)
          await fetch(`${LLAMA_BASE_URL}/models/config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model,
              config: {
                contextSize: activeModel.contextSize ?? 2048,
                gpuLayers: Number.isFinite(gpuLayers) ? gpuLayers : defaultGpuLayers,
              },
            }),
            signal: AbortSignal.timeout(2000),
          }).catch(() => {});

          // unload는 실패해도 무시(이미 안 떠있을 수 있음)
          await fetch(`${LLAMA_BASE_URL}/models/unload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
            signal: AbortSignal.timeout(2000),
          }).catch(() => {});

          await fetch(`${LLAMA_BASE_URL}/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model }),
            signal: AbortSignal.timeout(5000),
          }).catch(() => {});
        } catch (_e) {
          // 조용히 무시
        }
      }
    }

    alert('Settings saved successfully.');
    navigate('/');
  };

  const handleClose = () => navigate('/');

  const models = config?.models || [];
  const selectedModel = models.find(m => m.id === selectedModelId);
  const getModelLabel = (m) => (m?.modelPath || m?.name || m?.id || '').trim();
  
  const descriptionKeys = [
    "accelerator", "gpuLayers", "contextSize", "maxTokens", "temperature", 
    "topK", "topP", "minP", "tfsZ", "typicalP",
    "repeatPenalty", "repeatLastN", "presencePenalty", "frequencyPenalty", "penalizeNL",
    "dryMultiplier", "dryBase", "dryAllowedLength", "dryPenaltyLastN",
    "mirostatMode", "mirostatTau", "mirostatEta", "showSpecialTokens"
  ];

  return (
    <div className="settings-page-layout">
      <div className="settings-page-content">
        <PanelGroup direction="horizontal" className="settings-panel-group">
          <Panel defaultSize={30} minSize={15} maxSize={60}>
            <div className="model-list-panel">
              <h3>{t('settings.modelList')}</h3>
              <div className="model-list">
                {models.map(model => (
                  <div 
                    key={model.id} 
                    className={`model-list-item ${selectedModelId === model.id ? 'active' : ''}`}
                    onClick={() => handleSelectModel(model.id)}
                  >
                    <span className="model-name">{getModelLabel(model) || '-'}</span>
                    <button onClick={(e) => { e.stopPropagation(); handleDeleteModel(model.id); }} className="delete-button">
                      {t('settings.deleteModel')}
                    </button>
                  </div>
                ))}
              </div>
              <button onClick={handleAddNewModel} className="add-new-button">
                {t('settings.addNewModel')}
              </button>
            </div>
          </Panel>
          <PanelResizeHandle className="resizable-panel-handle" />
          <Panel minSize={30}>
            <div className="model-form-panel">
              {selectedModel ? (
                <ModelForm
                  key={selectedModel.id}
                  config={selectedModel}
                  onChange={handleModelFormChange}
                />
              ) : (
                <div className="no-model-selected">
                  <p>{t('settings.noModelSelected')}</p>
                </div>
              )}
              
              <div className="descriptions-section">
                <h3 
                  className="descriptions-header"
                  onClick={() => setIsDescriptionsExpanded(!isDescriptionsExpanded)}
                >
                  <span>{t('descriptions.title')}</span>
                  <span className={`expand-icon ${isDescriptionsExpanded ? 'expanded' : ''}`}>▼</span>
                </h3>
                {isDescriptionsExpanded && (
                  <div className="descriptions-grid">
                    {descriptionKeys.map(key => (
                      <DescriptionCard 
                        key={key}
                        title={t(`settings.${key}`)} 
                        content={t(`descriptions.${key}`)} 
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </Panel>
        </PanelGroup>
      </div>
      <div className="settings-footer">
        <button onClick={handleClose} className="footer-button close-button">{t('settings.close')}</button>
        <button onClick={handleSave} className="footer-button save-button">{t('settings.save')}</button>
      </div>
    </div>
  );
};

export default SettingsPage;
