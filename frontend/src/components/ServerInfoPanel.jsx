import React, { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import './ServerInfoPanel.css';

const ServerInfoPanel = () => {
  const { t } = useLanguage();
  const [config, setConfig] = useState(null);

  useEffect(() => {
    const loadConfig = () => {
      const defaultConfig = {
        name: 'Default Model',
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        minP: 0.05,
        tfsZ: 1.0,
        typicalP: 1.0,
        repeatPenalty: 1.1,
        repeatLastN: 64,
        penalizeNL: false,
        mirostatMode: 0,
        mirostatTau: 5.0,
        mirostatEta: 0.1,
        maxTokens: 1024,
      };
      const storedConfig = JSON.parse(localStorage.getItem('modelConfig'));
      setConfig({ ...defaultConfig, ...storedConfig });
    };

    loadConfig();
    window.addEventListener('storage', loadConfig);
    return () => window.removeEventListener('storage', loadConfig);
  }, []);

  if (!config) {
    return <div className="server-info-panel">Loading...</div>;
  }

  const mirostatModes = { 0: "Disabled", 1: "Mirostat v1", 2: "Mirostat v2" };

  return (
    <div className="server-info-panel">
      <h3>{t('infoPanel.title')}</h3>
      <div className="info-grid">
        <div className="info-item"><span className="info-label">{t('settings.temperature')}</span><span className="info-value">{config.temperature}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.topK')}</span><span className="info-value">{config.topK}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.topP')}</span><span className="info-value">{config.topP}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.minP')}</span><span className="info-value">{config.minP}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.repeatPenalty')}</span><span className="info-value">{config.repeatPenalty}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.repeatLastN')}</span><span className="info-value">{config.repeatLastN}</span></div>
        <div className="info-item"><span className="info-label">{t('settings.mirostatMode')}</span><span className="info-value">{mirostatModes[config.mirostatMode]}</span></div>
      </div>
       <div className="panel-footer">
        <p>{t('infoPanel.description')}</p>
      </div>
    </div>
  );
};

export default ServerInfoPanel;
