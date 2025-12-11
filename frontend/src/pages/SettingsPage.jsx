import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ModelForm from '../components/ModelForm';
import { useLanguage } from '../contexts/LanguageContext';
import './SettingsPage.css';

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
      }
    };
    loadConfig();
  }, []);

  const handleSelectModel = (modelId) => {
    setSelectedModelId(modelId);
  };
  
  const handleAddNewModel = () => {
    const newModel = {
      id: `model_${Date.now()}`,
      name: 'New Model',
      modelPath: '',
      accelerator: 'auto',
      gpuLayers: 0,
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
    const newModels = [...(config.models || []), newModel];
    setConfig({ ...config, models: newModels });
    setSelectedModelId(newModel.id);
  };
  
  const handleModelFormChange = (updatedModel) => {
    const newModels = (config.models || []).map(m => m.id === updatedModel.id ? updatedModel : m);
    setConfig({ ...config, models: newModels });
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
    if (window.electronAPI) {
      let configToSave = { ...(config || { models: [], activeModelId: null }) };
      if (!configToSave.activeModelId && configToSave.models && configToSave.models.length > 0) {
        configToSave.activeModelId = configToSave.models[0].id;
      }
      
      const result = await window.electronAPI.saveConfig(configToSave);
      if (result.success) {
        // 설정 저장 후 localStorage에 저장 (PerformancePanel이 읽을 수 있도록)
        const activeModel = configToSave.models.find(m => m.id === configToSave.activeModelId);
        if (activeModel) {
          localStorage.setItem('modelConfig', JSON.stringify(activeModel));
          
          // Context size 변경 이벤트 발생
          const contextSize = activeModel.contextSize || 2048;
          window.dispatchEvent(new CustomEvent('config-updated', {
            detail: { contextSize }
          }));
          
          // 설정 업데이트 이벤트 발생 (ChatPage에서 showSpecialTokens 읽기 위해)
          window.dispatchEvent(new CustomEvent('config-updated'));
        }
        
        alert('Settings saved successfully.');
        // 저장 성공 후 챗팅 창으로 이동
        navigate('/');
      } else {
        alert(`Failed to save settings: ${result.error}`);
      }
    }
  };

  const handleClose = () => navigate('/');

  const models = config?.models || [];
  const selectedModel = models.find(m => m.id === selectedModelId);
  
  const descriptionKeys = [
    "accelerator", "gpuLayers", "contextSize", "maxTokens", "temperature", 
    "topK", "topP", "minP", "tfsZ", "typicalP",
    "repeatPenalty", "repeatLastN", "presencePenalty", "frequencyPenalty", "penalizeNL",
    "dryMultiplier", "dryBase", "dryAllowedLength", "dryPenaltyLastN",
    "mirostatMode", "mirostatTau", "mirostatEta", "showSpecialTokens"
  ];

  return (
    <div className="settings-page-layout">
      <div className="model-list-panel">
        <h3>{t('settings.modelList')}</h3>
        <div className="model-list">
          {models.map(model => (
            <div 
              key={model.id} 
              className={`model-list-item ${selectedModelId === model.id ? 'active' : ''}`}
              onClick={() => handleSelectModel(model.id)}
            >
              <span className="model-name">{model.name}</span>
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
      <div className="settings-footer">
        <button onClick={handleClose} className="footer-button close-button">{t('settings.close')}</button>
        <button onClick={handleSave} className="footer-button save-button">{t('settings.save')}</button>
      </div>
    </div>
  );
};

export default SettingsPage;
