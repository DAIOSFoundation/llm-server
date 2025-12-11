import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Settings, LogIn } from 'lucide-react';
import { useLanguage } from '../contexts/LanguageContext';
import LanguageSelector from './LanguageSelector';
import ModelSelector from './ModelSelector';
import './Header.css';

const Header = () => {
  const navigate = useNavigate();
  const { t } = useLanguage();

  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title">LLM Server</h1>
        <ModelSelector />
      </div>
      <div className="header-right">
        <button
          className="header-button"
          onClick={() => navigate('/settings')}
          title={t('header.settings')}
        >
          <Settings size={20} />
        </button>
        <button
          className="header-button"
          onClick={() => {
            // 로그인 기능은 추후 구현
            alert('로그인 기능은 추후 구현 예정입니다.');
          }}
          title={t('header.login')}
        >
          <LogIn size={20} />
        </button>
        <LanguageSelector />
      </div>
    </header>
  );
};

export default Header;

