import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLanguage } from '../contexts/LanguageContext';
import { useAuth } from '../contexts/AuthContext';
import './LoginPage.css';

const isStrongPassword = (s) => {
  const v = String(s || '');
  if (v.length < 8) return false;
  const hasLower = /[a-z]/.test(v);
  const hasUpper = /[A-Z]/.test(v);
  const hasSpecial = /[^a-zA-Z0-9]/.test(v);
  return hasLower && hasUpper && hasSpecial;
};

const LoginPage = () => {
  const { t } = useLanguage();
  const nav = useNavigate();
  const auth = useAuth();

  const [mode, setMode] = useState('loading'); // loading | setup | login | verify
  const [superAdminId, setSuperAdminId] = useState('');
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [busy, setBusy] = useState(false);

  const superIdOk = String(superAdminId || '').trim().length > 0;
  const pwPolicyOk = isStrongPassword(password);
  const pwConfirmFilled = Boolean(password2);
  const pwMatch = password && password2 && password === password2;

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
      setSuperAdminId('superadmin');
      setMode('setup');
    }
  }, [auth?.loading, auth?.authenticated, auth?.initialized, auth?.superAdminId]);

  const onSetup = async (e) => {
    e.preventDefault();
    if (!isStrongPassword(password)) {
      window.alert(t('login.passwordPolicy'));
      return;
    }
    if (password !== password2) {
      window.alert(t('login.passwordMismatch'));
      return;
    }
    if (!superIdOk) {
      window.alert(t('login.superAdminIdRequired'));
      return;
    }
    setBusy(true);
    try {
      await auth.setup({ superAdminId, password });
      setPassword('');
      setPassword2('');
      // If setup succeeded, the account exists now -> show login screen.
      setMode('login');
    } catch (_err) {
      window.alert(t('login.setupFailed'));
    } finally {
      setBusy(false);
    }
  };

  const onLogin = async (e) => {
    e.preventDefault();
    if (!password) return;
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
              <input
                value={superAdminId}
                onChange={(e) => setSuperAdminId(e.target.value)}
                placeholder={t('login.superAdminIdPlaceholder')}
                autoFocus
              />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
                autoFocus
              />
            </div>
            <div className="login-row">
              <label>{t('login.passwordConfirm')}</label>
              <input
                type="password"
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
              />
            </div>
            {!pwPolicyOk && password ? <div className="login-hint">{t('login.passwordPolicy')}</div> : null}
            {pwConfirmFilled && !pwMatch ? <div className="login-error">{t('login.passwordMismatch')}</div> : null}
            <button className="login-button" type="submit" disabled={busy}>
              {busy ? t('login.working') : t('login.create')}
            </button>
          </form>
        ) : null}

        {mode === 'verify' ? (
          <form className="login-form" onSubmit={onLogin}>
            <div className="login-note">{t('login.verifyNote')}</div>
            <div className="login-row">
              <label>{t('login.superAdminId')}</label>
              <input
                value={superAdminId}
                onChange={(e) => setSuperAdminId(e.target.value)}
                placeholder={t('login.superAdminIdPlaceholder')}
              />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
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
              <input
                value={superAdminId}
                onChange={(e) => setSuperAdminId(e.target.value)}
                placeholder={t('login.superAdminIdPlaceholder')}
              />
            </div>
            <div className="login-row">
              <label>{t('login.password')}</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={t('login.passwordPlaceholder')}
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

