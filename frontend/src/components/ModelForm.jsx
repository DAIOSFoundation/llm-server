import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './ModelForm.css';

const ModelForm = ({ config, onChange }) => {
  const { t } = useLanguage();
  const [formData, setFormData] = useState({});

  useEffect(() => {
    // The form is now fully controlled by the parent component's state
    // We add default values here to prevent uncontrolled component warnings
    const defaults = {
      name: 'New Model', modelPath: '', temperature: 0.7, topK: 40, topP: 0.95,
      minP: 0.05, tfsZ: 1.0, typicalP: 1.0, repeatPenalty: 1.1, repeatLastN: 64,
      penalizeNL: false, mirostatMode: 0, mirostatTau: 5.0, mirostatEta: 0.1, maxTokens: 1024,
    };
    setFormData({ ...defaults, ...config });
  }, [config]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newFormData = {
      ...formData,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseFloat(value) : value),
    };
    setFormData(newFormData);
    onChange(newFormData); // Notify parent of the change
  };

  const handleFindModel = async () => {
    if (window.electronAPI) {
      const filePath = await window.electronAPI.selectFile();
      if (filePath) {
        const newFormData = { ...formData, modelPath: filePath };
        setFormData(newFormData);
        onChange(newFormData);
      }
    } else {
      console.error('Electron API is not available.');
    }
  };
  
  // A special handler just for the model name, as it's not a standard input
  const handleNameChange = (e) => {
    const newFormData = { ...formData, name: e.target.value };
    setFormData(newFormData);
    onChange(newFormData);
  }

  return (
    <div className="model-form">
      {/* Model Name Section */}
      <div className="form-section">
        <h3>{t('settings.modelName')}</h3>
        <div className="form-group">
           <input 
            type="text" 
            name="name" 
            value={formData.name || ''} 
            onChange={handleNameChange}
            className="model-name-input"
          />
        </div>
      </div>
      
      {/* Model Path Section */}
      <div className="form-section">
        <h3>{t('settings.modelPath')}</h3>
        <div className="form-group model-path-group">
          <input 
            type="text" 
            name="modelPath" 
            value={formData.modelPath || ''} 
            readOnly 
            placeholder="No model selected"
          />
          <button type="button" onClick={handleFindModel} className="find-button">
            {t('settings.findModel')}
          </button>
        </div>
      </div>
      
      {/* Inference, Sampling, Penalties, Mirostat sections... */}
            <div className="form-section">
        <h3>{t('settings.inference')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.maxTokens')}</label>
            <input type="number" name="maxTokens" value={formData.maxTokens} onChange={handleChange} min="1" />
          </div>
           <div className="form-group">
            <label>{t('settings.temperature')}</label>
            <input type="number" name="temperature" value={formData.temperature} onChange={handleChange} step="0.01" min="0" />
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>{t('settings.sampling')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.topK')}</label>
            <input type="number" name="topK" value={formData.topK} onChange={handleChange} min="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.topP')}</label>
            <input type="number" name="topP" value={formData.topP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.minP')}</label>
            <input type="number" name="minP" value={formData.minP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
          <div className="form-group">
            <label>{t('settings.tfsZ')}</label>
            <input type="number" name="tfsZ" value={formData.tfsZ} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.typicalP')}</label>
            <input type="number" name="typicalP" value={formData.typicalP} onChange={handleChange} step="0.01" min="0" max="1" />
          </div>
        </div>
      </div>
      
      <div className="form-section">
        <h3>{t('settings.penalties')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.repeatPenalty')}</label>
            <input type="number" name="repeatPenalty" value={formData.repeatPenalty} onChange={handleChange} step="0.01" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.repeatLastN')}</label>
            <input type="number" name="repeatLastN" value={formData.repeatLastN} onChange={handleChange} min="0" />
          </div>
          <div className="form-group checkbox-group">
            <label htmlFor="penalizeNL">{t('settings.penalizeNL')}</label>
            <input id="penalizeNL" type="checkbox" name="penalizeNL" checked={formData.penalizeNL} onChange={handleChange} />
          </div>
        </div>
      </div>
      
      <div className="form-section">
        <h3>{t('settings.mirostat')}</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>{t('settings.mirostatMode')}</label>
             <select name="mirostatMode" value={formData.mirostatMode} onChange={handleChange}>
              <option value="0">Disabled</option>
              <option value="1">Mirostat v1</option>
              <option value="2">Mirostat v2</option>
            </select>
          </div>
          <div className="form-group">
            <label>{t('settings.mirostatTau')}</label>
            <input type="number" name="mirostatTau" value={formData.mirostatTau} onChange={handleChange} step="0.1" min="0" />
          </div>
          <div className="form-group">
            <label>{t('settings.mirostatEta')}</label>
            <input type="number" name="mirostatEta" value={formData.mirostatEta} onChange={handleChange} step="0.01" min="0" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelForm;
