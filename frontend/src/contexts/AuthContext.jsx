import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { LLAMA_BASE_URL } from '../services/api';

// 인증 서버는 별도 포트(8082)에서 실행
const AUTH_BASE_URL = typeof window !== 'undefined' && window.location.hostname === 'localhost'
  ? 'http://localhost:8082'
  : (LLAMA_BASE_URL || 'http://localhost:8080').replace(':8080', ':8082');

const AuthContext = createContext(null);

const TOKEN_KEY = 'llmServerUiAuthToken';

const getToken = () => {
  try {
    return String(localStorage.getItem(TOKEN_KEY) || '');
  } catch (_e) {
    return '';
  }
};

const setToken = (token) => {
  try {
    if (!token) {
      localStorage.removeItem(TOKEN_KEY);
    } else {
      localStorage.setItem(TOKEN_KEY, token);
    }
  } catch (_e) {
    // ignore
  }
};

export const AuthProvider = ({ children }) => {
  const [initialized, setInitialized] = useState(false);
  const [superAdminId, setSuperAdminId] = useState('');
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const refreshInFlightRef = useRef(false);

  const refreshStatus = async () => {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    setLoading(true);
    try {
      const token = getToken();
      const res = await fetch(`${AUTH_BASE_URL}/auth/status`, {
        method: 'GET',
        headers: token ? { 'X-LLM-UI-Auth': token } : {},
        // Router mode can be busy (model load / IO). Avoid flipping UI to "not initialized"
        // due to transient timeouts.
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const json = await res.json();
        setInitialized(Boolean(json.initialized));
        setSuperAdminId(String(json.superAdminId || ''));
        setAuthenticated(Boolean(json.authenticated));
      } else {
        // Keep previous initialized state on non-OK responses.
        // We only reset authenticated because a failed status check means "not logged in".
        setAuthenticated(false);
      }
    } catch (_e) {
      // Keep previous initialized/superAdminId on transient network failures.
      setAuthenticated(false);
    } finally {
      setLoading(false);
      refreshInFlightRef.current = false;
    }
  };

  const setup = async ({ superAdminId: id, password }) => {
    const res = await fetch(`${AUTH_BASE_URL}/auth/setup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ superAdminId: id, password }),
      signal: AbortSignal.timeout(5000),
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.ok) {
      throw new Error(json?.error || 'setup_failed');
    }
    await refreshStatus();
    return json;
  };

  const login = async ({ superAdminId: id, password }) => {
    const res = await fetch(`${AUTH_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ superAdminId: id, password }),
      signal: AbortSignal.timeout(5000),
    });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.ok || !json.token) {
      throw new Error(json?.error || 'login_failed');
    }
    setToken(String(json.token));
    await refreshStatus();
    return json;
  };

  const logout = async () => {
    try {
      const token = getToken();
      await fetch(`${AUTH_BASE_URL}/auth/logout`, {
        method: 'POST',
        headers: token ? { 'X-LLM-UI-Auth': token } : {},
        signal: AbortSignal.timeout(2000),
      }).catch(() => {});
    } finally {
      setToken('');
      setAuthenticated(false);
      setSuperAdminId('');
      // keep initialized state by reloading status (it may still be initialized)
      await refreshStatus();
    }
  };

  const value = useMemo(
    () => ({
      initialized,
      superAdminId,
      authenticated,
      loading,
      refreshStatus,
      setup,
      login,
      logout,
    }),
    [initialized, superAdminId, authenticated, loading],
  );

  useEffect(() => {
    refreshStatus();
    const onFocus = () => refreshStatus();
    const onVisibility = () => {
      if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
        refreshStatus();
      }
    };
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onVisibility);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, []);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => useContext(AuthContext);

