import React, { useState, useEffect } from 'react';
import { tokenizeText } from '../services/api';
import './TokenDebugPanel.css';

const TokenDebugPanel = () => {
  const [promptTokens, setPromptTokens] = useState([]);
  const [responseTokens, setResponseTokens] = useState([]);

  useEffect(() => {
    // 공통 normalize 함수
    const normalizeTokens = (tokenData) => {
      return tokenData.map((t, index) => {
        // with_pieces=true 일 때: { id, piece }
        let id = t.id ?? t;
        let piece = '';
        if (typeof t.piece === 'string') {
          piece = t.piece;
        } else if (Array.isArray(t.piece)) {
          // 바이트 배열일 경우 문자열로 변환 시도
          try {
            piece = String.fromCharCode(...t.piece);
          } catch {
            piece = JSON.stringify(t.piece);
          }
        } else if (typeof t === 'string') {
          piece = t;
        }
        const isSpecial = /^<\|[^>]*\|>$/.test(piece);
        return { index, id, piece, isSpecial };
      });
    };

    const handlePromptOutput = async (event) => {
      const prompt = event.detail?.prompt || '';
      if (!prompt) {
        setPromptTokens([]);
        return;
      }
      try {
        const tokenData = await tokenizeText(prompt);
        const normalized = normalizeTokens(tokenData);
        setPromptTokens(normalized);
      } catch (error) {
        console.error('Failed to tokenize prompt:', error);
      }
    };

    const handleAssistantOutput = async (event) => {
      const text = event.detail?.text || '';
      if (!text) {
        setResponseTokens([]);
        return;
      }
      try {
        const tokenData = await tokenizeText(text);
        const normalized = normalizeTokens(tokenData);
        setResponseTokens(normalized);
      } catch (error) {
        console.error('Failed to tokenize assistant output:', error);
      }
    };

    window.addEventListener('prompt-output', handlePromptOutput);
    window.addEventListener('assistant-output', handleAssistantOutput);
    return () => {
      window.removeEventListener('prompt-output', handlePromptOutput);
      window.removeEventListener('assistant-output', handleAssistantOutput);
    };
  }, []);

  const renderTokenSequence = (tokens) => {
    if (!tokens.length) return null;
    return (
      <div className="token-sequence">
        {tokens.map((t, i) => (
          <span
            key={i}
            className={t.isSpecial ? 'special-token' : 'normal-token'}
          >
            {t.piece}
          </span>
        ))}
      </div>
    );
  };

  const hasAnyTokens = promptTokens.length > 0 || responseTokens.length > 0;

  return (
    <div className="token-debug-panel">
      <h4>Token Debug (Prompt & Response)</h4>
      {!hasAnyTokens && (
        <div className="token-debug-empty">
          아직 토큰 정보가 없습니다. LLM 응답을 생성하면 여기에서 토큰을 확인할 수 있습니다.
        </div>
      )}
      {hasAnyTokens && (
        <>
          {promptTokens.length > 0 && (
            <div className="token-debug-section">
              {renderTokenSequence(promptTokens)}
            </div>
          )}
          {responseTokens.length > 0 && (
            <div className="token-debug-section">
              <div className="token-debug-label">Response (토큰 순서 그대로)</div>
              {renderTokenSequence(responseTokens)}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default TokenDebugPanel;
