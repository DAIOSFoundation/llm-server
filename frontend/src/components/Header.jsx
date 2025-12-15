import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import LanguageSelector from './LanguageSelector';
import { useLanguage } from '../contexts/LanguageContext';
import './Header.css';

const Header = () => {
  const { t } = useLanguage();
  const [config, setConfig] = useState({ models: [], activeModelId: null });
  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const STORAGE_KEY = 'llmServerClientConfig';

  useEffect(() => {
    const loadConfig = async () => {
      if (window.electronAPI) {
        const loadedConfig = await window.electronAPI.loadConfig();
        // Ensure loadedConfig and its models property are not null/undefined
        if (loadedConfig && loadedConfig.models) {
          setConfig(loadedConfig);
        }
      } else {
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) {
            const parsed = JSON.parse(raw);
            if (parsed && parsed.models) {
              setConfig(parsed);
            }
          }
        } catch (_e) {
          // ignore
        }
      }
    };
    loadConfig();
    
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  const handleSelectModel = async (modelId) => {
    const newConfig = { ...config, activeModelId: modelId };
    setConfig(newConfig);
    setDropdownOpen(false);
    if (window.electronAPI) {
      await window.electronAPI.saveConfig(newConfig); // This will also restart the server
    } else {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));
      } catch (_e) {
        // ignore
      }
    }

    // 클라이언트 추론 파라미터용 active model도 갱신
    const models = newConfig?.models || [];
    const activeModel = models.find(m => m.id === newConfig?.activeModelId);
    if (activeModel) {
      try {
        localStorage.setItem('modelConfig', JSON.stringify(activeModel));
        const contextSize = activeModel.contextSize || 2048;
        window.dispatchEvent(new CustomEvent('config-updated', { detail: { contextSize } }));
        window.dispatchEvent(new CustomEvent('config-updated'));
      } catch (_e) {
        // ignore
      }
    }
  };

  // Defensive coding: ensure config and config.models exist before trying to use them
  const models = config?.models || [];
  const activeModel = models.find(m => m.id === config?.activeModelId);

  return (
    <header className="app-header">
      <div className="header-left">
        <h1>LLM Server</h1>
        <nav>
          <NavLink to="/" className={({ isActive }) => (isActive ? 'active' : '')}>
            Chat
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}>
            {t('header.settings')}
          </NavLink>
        </nav>
      </div>

      <div className="header-center" ref={dropdownRef}>
        <div className="model-selector-dropdown">
          <button className="current-model-display" onClick={() => setDropdownOpen(!isDropdownOpen)}>
            {activeModel ? activeModel.name : t('header.selectModel')}
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
                  {model.name}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="header-right">
        <LanguageSelector />
        {/* Login button can be added here */}
      </div>
    </header>
  );
};

export default Header;
