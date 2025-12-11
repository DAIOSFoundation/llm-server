import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ModelForm from '../components/ModelForm';
import { useLanguage } from '../contexts/LanguageContext';
import './SettingsPage.css';

const SettingsPage = () => {
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [config, setConfig] = useState({ models: [], activeModelId: null });
  const [selectedModelId, setSelectedModelId] = useState(null);

  useEffect(() => {
    const loadConfig = async () => {
      if (window.electronAPI) {
        const loadedConfig = await window.electronAPI.loadConfig();
        if (loadedConfig && loadedConfig.models) {
          setConfig(loadedConfig);
          if (loadedConfig.activeModelId) {
            setSelectedModelId(loadedConfig.activeModelId);
          } else if (loadedConfig.models.length > 0) {
            // If no active model is set, select the first one by default
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
      temperature: 0.7,
      topK: 40,
      topP: 0.95,
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
        alert('Settings saved successfully.');
      } else {
        alert(`Failed to save settings: ${result.error}`);
      }
    }
  };

  const handleClose = () => navigate('/');

  const models = config?.models || [];
  const selectedModel = models.find(m => m.id === selectedModelId);

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
      </div>
      <div className="settings-footer">
        <button onClick={handleClose} className="footer-button close-button">{t('settings.close')}</button>
        <button onClick={handleSave} className="footer-button save-button">{t('settings.save')}</button>
      </div>
    </div>
  );
};

export default SettingsPage;
