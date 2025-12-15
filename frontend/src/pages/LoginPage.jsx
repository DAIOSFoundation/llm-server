import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../contexts/AuthContext';
import './LoginPage.css';

const randomId = () => {
  const part = Math.random().toString(16).slice(2, 10);
  return `superadmin_${part}`;
};

const is8Digits = (s) => /^\d{8}$/.test(String(s || ''));

const LoginPage = () => {
  const { t } = useLanguage();
  const nav = useNavigate();
  const auth = useAuth();

  const initialSuperId = useMemo(() => randomId(), []);
  const [mode, setMode] = useState('loading'); // loading | setup | login | verify
  const [superAdminId, setSuperAdminId] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!auth) return;
    if (auth.loading) {
      setMode('loading');
      return;
    }
    if (auth.authenticated) {
      nav('/');
      return;
    }
    if (auth.initialized) {
      setSuperAdminId(auth.superAdminId || '');
      setMode('login');
    } else {
      setSuperAdminId(initialSuperId);
      setMode('setup');
    }
  }, [auth?.loading, auth?.authenticated, auth?.initialized, auth?.superAdminId]);

  const onSetup = async (e) => {
    e.preventDefault();
    if (!is8Digits(password)) {
      window.alert(t('login.password8digits'));
      return;
    }
    if (password !== password2) {
      window.alert(t('login.passwordMismatch'));
      return;
    }
    setBusy(true);
    try {
      await auth.setup({ superAdminId, password });
      setPassword('');
      setPassword2('');
      setMode('verify'); // 검증 화면
    } catch (_err) {
      window.alert(t('login.setupFailed'));
    } finally {
      setBusy(false);
    }
  };

  const onLogin = async (e) => {
    e.preventDefault();
    if (!is8Digits(password)) {
      window.alert(t('login.password8digits'));
      return;
    }
    setBusy(true);
    try {
      await auth.login({ superAdminId, password });
      nav('/');
    } catch (_err) {
      window.alert(`${t('login.failAlert')}\n\n${t('login.contact')}: tony@banya.ai`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <h2 className="login-title">{t('login.title')}</h2>
        <p className="login-subtitle">{t('login.subtitle')}</p>

        {mode === 'loading' ? (
          <div className="login-loading">{t('login.loading')}</div>
        ) : null}

        {mode === 'setup' ? (
          <form className="login-form" onSubmit={onSetup}>
            <div className="login-row">
              <label>{t('login.superAdminId')}</label>
              <input value={superAdminId} readOnly />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
                inputMode="numeric"
                maxLength={8}
                autoFocus
              />
            </div>
            <div className="login-row">
              <label>{t('login.passwordConfirm')}</label>
              <input
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
                inputMode="numeric"
                maxLength={8}
              />
            </div>
            <button className="login-button" type="submit" disabled={busy}>
              {busy ? t('login.working') : t('login.setup')}
            </button>
          </form>
        ) : null}

        {mode === 'verify' ? (
          <form className="login-form" onSubmit={onLogin}>
            <div className="login-note">{t('login.verifyNote')}</div>
            <div className="login-row">
              <label>{t('login.superAdminId')}</label>
              <input value={superAdminId} readOnly />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
                inputMode="numeric"
                maxLength={8}
                autoFocus
              />
            </div>
            <button className="login-button" type="submit" disabled={busy}>
              {busy ? t('login.working') : t('login.login')}
            </button>
          </form>
        ) : null}

        {mode === 'login' ? (
          <form className="login-form" onSubmit={onLogin}>
            <div className="login-row">
              <label>{t('login.superAdminId')}</label>
              <input value={superAdminId} readOnly />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
                inputMode="numeric"
                maxLength={8}
                autoFocus
              />
            </div>
            <button className="login-button" type="submit" disabled={busy}>
              {busy ? t('login.working') : t('login.login')}
            </button>
          </form>
        ) : null}
      </div>
    </div>
  );
};

export default LoginPage;

