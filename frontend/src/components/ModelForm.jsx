import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './ModelForm.css';

const ModelForm = ({ model, onSubmit, onCancel }) => {
  const { t } = useLanguage();
  const [formData, setFormData] = useState({
    name: '',
    path: '',
    quantization: 'Q4_0',
    device: 'CPU',
    gpuBackend: '',
    inferenceConfig: {
      contextSize: 2048,
      batchSize: 512,
      maxTokens: 512,
      temperature: 0.7,
      topK: 40,
      topP: 0.9,
      repeatPenalty: 1.1,
      repeatLastN: 64,
      threads: 4,
      threadsBatch: 4,
      seed: -1,
      stopSequences: ''
    }
  });

  useEffect(() => {
    if (model) {
      setFormData({
        name: model.name || '',
        path: model.path || '',
        quantization: model.quantization || 'Q4_0',
        device: model.device || 'CPU',
        gpuBackend: model.gpuBackend || '',
        inferenceConfig: {
          ...formData.inferenceConfig,
          ...model.inferenceConfig
        }
      });
    }
  }, [model]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (name.startsWith('inference.')) {
      const key = name.replace('inference.', '');
      setFormData(prev => ({
        ...prev,
        inferenceConfig: {
          ...prev.inferenceConfig,
          [key]: type === 'number' ? parseFloat(value) : value
        }
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: type === 'checkbox' ? checked : value
      }));
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFormData(prev => ({
        ...prev,
        path: file.path || file.name
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const submitData = {
      ...formData,
      inferenceConfig: {
        ...formData.inferenceConfig,
        stopSequences: formData.inferenceConfig.stopSequences
          .split(',')
          .map(s => s.trim())
          .filter(s => s)
      }
    };

    onSubmit(submitData);
  };

  const quantizationOptions = [
    'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'F16', 'F32'
  ];

  return (
    <form className="model-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <h3>기본 정보</h3>
        <div className="form-group">
          <label>{t('settings.modelName')}</label>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
          />
        </div>
        <div className="form-group">
          <label>{t('settings.modelPath')}</label>
          <div className="file-input-group">
            <input
              type="text"
              name="path"
              value={formData.path}
              onChange={handleChange}
              placeholder="모델 파일 경로"
              required
            />
            <input
              type="file"
              id="file-select"
              accept=".gguf"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <button
              type="button"
              onClick={() => document.getElementById('file-select').click()}
              className="file-select-button"
            >
              {t('settings.selectFile')}
            </button>
          </div>
        </div>
        <div className="form-group">
          <label>{t('settings.quantization')}</label>
          <select
            name="quantization"
            value={formData.quantization}
            onChange={handleChange}
          >
            {quantizationOptions.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>{t('settings.device')}</label>
          <select
            name="device"
            value={formData.device}
            onChange={handleChange}
          >
            <option value="CPU">CPU</option>
            <option value="GPU">GPU</option>
          </select>
        </div>
        {formData.device === 'GPU' && (
          <div className="form-group">
            <label>{t('settings.gpuBackend')}</label>
            <select
              name="gpuBackend"
              value={formData.gpuBackend}
              onChange={handleChange}
            >
              <option value="">자동 선택</option>
              <option value="CUDA">CUDA</option>
              <option value="OpenCL">OpenCL</option>
            </select>
          </div>
        )}
      </div>

      <div className="form-section">
        <h3>{t('settings.inference')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.contextSize')}</label>
            <input
              type="number"
              name="inference.contextSize"
              value={formData.inferenceConfig.contextSize}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.batchSize')}</label>
            <input
              type="number"
              name="inference.batchSize"
              value={formData.inferenceConfig.batchSize}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.maxTokens')}</label>
            <input
              type="number"
              name="inference.maxTokens"
              value={formData.inferenceConfig.maxTokens}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.temperature')}</label>
            <input
              type="number"
              name="inference.temperature"
              value={formData.inferenceConfig.temperature}
              onChange={handleChange}
              min="0"
              max="2"
              step="0.1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.topK')}</label>
            <input
              type="number"
              name="inference.topK"
              value={formData.inferenceConfig.topK}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.topP')}</label>
            <input
              type="number"
              name="inference.topP"
              value={formData.inferenceConfig.topP}
              onChange={handleChange}
              min="0"
              max="1"
              step="0.01"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.repeatPenalty')}</label>
            <input
              type="number"
              name="inference.repeatPenalty"
              value={formData.inferenceConfig.repeatPenalty}
              onChange={handleChange}
              min="0"
              step="0.1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.repeatLastN')}</label>
            <input
              type="number"
              name="inference.repeatLastN"
              value={formData.inferenceConfig.repeatLastN}
              onChange={handleChange}
              min="0"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.threads')}</label>
            <input
              type="number"
              name="inference.threads"
              value={formData.inferenceConfig.threads}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.threadsBatch')}</label>
            <input
              type="number"
              name="inference.threadsBatch"
              value={formData.inferenceConfig.threadsBatch}
              onChange={handleChange}
              min="1"
            />
          </div>
          <div className="form-group">
            <label>{t('settings.seed')}</label>
            <input
              type="number"
              name="inference.seed"
              value={formData.inferenceConfig.seed}
              onChange={handleChange}
            />
          </div>
          <div className="form-group">
            <label>{t('settings.stopSequences')}</label>
            <input
              type="text"
              name="inference.stopSequences"
              value={formData.inferenceConfig.stopSequences}
              onChange={handleChange}
              placeholder="쉼표로 구분"
            />
          </div>
        </div>
      </div>

      <div className="form-actions">
        <button type="button" onClick={onCancel} className="cancel-button">
          {t('settings.cancel')}
        </button>
        <button type="submit" className="submit-button">
          {t('settings.save')}
        </button>
      </div>
    </form>
  );
};

export default ModelForm;

