import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { getModels } from '../services/api';
import './ModelSelector.css';

const ModelSelector = () => {
  const { t } = useLanguage();
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState(() => {
    return localStorage.getItem('selectedModelId') || '';
  });

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModelId) {
      localStorage.setItem('selectedModelId', selectedModelId);
    }
  }, [selectedModelId]);

  const loadModels = async () => {
    try {
      const data = await getModels();
      if (data.success) {
        setModels(data.models);
        if (data.models.length > 0 && !selectedModelId) {
          setSelectedModelId(data.models[0].id);
        }
      }
    } catch (error) {
      console.error('모델 로드 실패:', error);
    }
  };

  const handleModelChange = (e) => {
    setSelectedModelId(e.target.value);
  };

  if (models.length === 0) {
    return (
      <div className="model-selector-empty">
        {t('chat.noModel')}
      </div>
    );
  }

  return (
    <select
      className="model-selector"
      value={selectedModelId}
      onChange={handleModelChange}
    >
      {models.map((model) => (
        <option key={model.id} value={model.id}>
          {model.name}
        </option>
      ))}
    </select>
  );
};

export default ModelSelector;

