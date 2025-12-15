import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { LanguageProvider } from './contexts/LanguageContext';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Header from './components/Header';
import ChatPage from './pages/ChatPage';
import SettingsPage from './pages/SettingsPage';
import GuidePage from './pages/GuidePage';
import LoginPage from './pages/LoginPage';
import './App.css';

const RequireAuth = ({ children }) => {
  const auth = useAuth();
  if (!auth) return children;
  if (auth.loading) return null;
  if (!auth.authenticated) {
    return <LoginPage />;
  }
  return children;
};

function App() {
  return (
    <LanguageProvider>
      <AuthProvider>
        <Router>
          <div className="App">
            <Header />
            <main className="main-content">
              <Routes>
                <Route path="/login" element={<LoginPage />} />
                <Route
                  path="/"
                  element={
                    <RequireAuth>
                      <ChatPage />
                    </RequireAuth>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <RequireAuth>
                      <SettingsPage />
                    </RequireAuth>
                  }
                />
                <Route
                  path="/guide"
                  element={
                    <RequireAuth>
                      <GuidePage />
                    </RequireAuth>
                  }
                />
              </Routes>
            </main>
          </div>
        </Router>
      </AuthProvider>
    </LanguageProvider>
  );
}

export default App;

