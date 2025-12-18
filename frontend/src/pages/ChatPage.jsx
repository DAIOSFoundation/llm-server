import React, { useState, useEffect, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { sendChatMessage, buildLlama3Prompt, countTokens, getActiveServerUrl } from '../services/api';
import { useLanguage } from '../contexts/LanguageContext';
import LogPanel from '../components/LogPanel';
import ProgressBar from '../components/ProgressBar';
import './ChatPage.css';

const ChatPage = () => {
  const { t, language } = useLanguage();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [showSpecialTokens, setShowSpecialTokens] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const assistantOutputRef = useRef('');
  const streamBufferRef = useRef('');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  // Context 사용량 계산 및 업데이트
  useEffect(() => {
    const calculateContextUsage = async () => {
      try {
        const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
        const contextSize = config.contextSize || 2048;
        
        // 실제 프롬프트를 생성하여 정확한 토큰 수 계산
        const prompt = buildLlama3Prompt(messages, language);
        const promptLength = prompt.length;
        
        // 서버의 tokenize API를 사용하여 정확한 토큰 수 계산
        const actualTokens = await countTokens(prompt);
        
        // Context 사용량 이벤트 발생
        window.dispatchEvent(new CustomEvent('context-update', {
          detail: {
            used: actualTokens,
            total: contextSize
          }
        }));
        
        // 경고 메시지 출력 (디버그용 로그는 주석 처리)
        // console.log(`[Context] Prompt length: ${promptLength} chars, Actual tokens: ${actualTokens}, Context size: ${contextSize}`);
        if (actualTokens > contextSize) {
          console.error(`[Context] Error: Actual tokens (${actualTokens}) exceeds context size (${contextSize})`);
        } else if (actualTokens > contextSize * 0.9) {
          console.warn(`[Context] Warning: Actual tokens (${actualTokens}) is close to context size (${contextSize})`);
        }
      } catch (error) {
        console.error('Failed to calculate context usage:', error);
        // 에러 발생 시 추정값 사용
        try {
          const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
          const contextSize = config.contextSize || 2048;
          const prompt = buildLlama3Prompt(messages, language);
          const estimatedTokens = Math.ceil(prompt.length / 2.0);
          window.dispatchEvent(new CustomEvent('context-update', {
            detail: {
              used: estimatedTokens,
              total: contextSize
            }
          }));
        } catch (e) {
          console.error('Failed to use fallback estimation:', e);
        }
      }
    };

    calculateContextUsage();
  }, [messages, language]);

  // 스페셜 토큰 표시 설정 로드
  useEffect(() => {
    const loadShowSpecialTokens = () => {
      try {
        const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
        setShowSpecialTokens(config.showSpecialTokens || false);
      } catch (error) {
        console.error('Failed to load showSpecialTokens setting:', error);
      }
    };
    loadShowSpecialTokens();

    // 설정 변경 이벤트 리스너
    const handleConfigUpdate = () => {
      loadShowSpecialTokens();
    };
    window.addEventListener('config-updated', handleConfigUpdate);

    return () => {
      window.removeEventListener('config-updated', handleConfigUpdate);
    };
  }, []);

  // 모델 로딩 상태 체크 및 서버 자동 시작
  useEffect(() => {
    let retryCount = 0;
    const MAX_RETRIES = 3;
    let hasRequestedStart = false;

    const checkServerStatus = async () => {
      try {
        // 라우터 모드 포함: /health 로 서버 기동 여부만 체크
        const serverUrl = getActiveServerUrl();
        const response = await fetch(`${serverUrl}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(1000),
        });
        if (response.ok) {
          const data = await response.json().catch(() => ({}));
          // status가 "loading"이면 모델이 아직 로딩 중
          if (data.status === 'loading') {
            setIsModelLoading(true);
          } else if (data.status === 'ready') {
            setIsModelLoading(false);
            retryCount = 0; // 성공 시 리트라이 카운트 리셋
            hasRequestedStart = false; // 성공 시 플래그 리셋
          } else {
            // 기본적으로 ready로 간주
            setIsModelLoading(false);
          }
          return;
        }
      } catch (error) {
        // 서버가 아직 시작되지 않았거나 연결 불가
        if (error.name === 'AbortError' || error.message.includes('Failed to fetch') || error.message.includes('ERR_CONNECTION_REFUSED')) {
          // 서버가 없는 경우, 클라이언트 모드에서 서버 시작 요청
          if (!window.electronAPI && !hasRequestedStart && retryCount < MAX_RETRIES) {
            retryCount++;
            hasRequestedStart = true;
            
            try {
              // 클라이언트 모드: 서버 시작 요청
              const configStr = localStorage.getItem('llmServerClientConfig');
              if (configStr) {
                const config = JSON.parse(configStr);
                if (config.activeModelId) {
                  console.log('[ChatPage] Server not running, requesting server start for model:', config.activeModelId);
                  // /api/save-config를 호출하여 서버 시작 트리거
                  await fetch('http://localhost:8083/api/save-config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: configStr,
                    signal: AbortSignal.timeout(2000),
                  }).catch(() => {
                    // 실패해도 조용히 처리
                  });
                }
              }
            } catch (e) {
              // 에러는 조용히 처리
              console.log('[ChatPage] Failed to request server start:', e.message);
            }
          }
        }
      }
    };

    // 모델 로딩 이벤트 리스너
    const handleModelLoading = (event) => {
      setIsModelLoading(event.detail.loading);
    };

    window.addEventListener('model-loading', handleModelLoading);

    // 초기 체크 (즉시)
    checkServerStatus();

    // 주기적으로 체크 (2초마다, 서버 시작 후에는 1초마다)
    const interval = setInterval(checkServerStatus, 2000);

    return () => {
      window.removeEventListener('model-loading', handleModelLoading);
      clearInterval(interval);
    };
  }, []);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;

    const userMessage = { role: 'user', content: input };
    const nextMessages = [...messages, userMessage];

    // 디버그용: 현재 프롬프트 전체를 Token Debug 패널로 전달
    try {
      const fullPrompt = buildLlama3Prompt(nextMessages, language);
      // console.log('[ChatPage] Dispatching prompt-output event, prompt length:', fullPrompt.length);
      window.dispatchEvent(new CustomEvent('prompt-output', {
        detail: { prompt: fullPrompt },
      }));
    } catch (e) {
      // console.error('Failed to build prompt for debug:', e);
    }

    // 새 응답 시작 시 이전 assistant 출력 및 버퍼 초기화
    assistantOutputRef.current = '';
    streamBufferRef.current = '';

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsWaitingForFirstToken(true);

    const assistantMessage = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      await sendChatMessage([...messages, userMessage], (token) => {
        // 첫 토큰이 수신되면 로딩 애니메이션 숨김
        if (isWaitingForFirstToken) {
          setIsWaitingForFirstToken(false);
        }
        // 토큰이 수신되면 모델 로딩 완료
        setIsModelLoading(false);
        // 디버깅용으로 전체 assistant 출력 누적
        assistantOutputRef.current += token;
        
        // 즉시 화면에 반영 (서버에서 이미 올바른 증분 텍스트를 보내고 있음)
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage && lastMessage.role === 'assistant') {
            // 서버에서 이미 새로운 부분만 보내고 있으므로 바로 추가
            return [
              ...prev.slice(0, -1),
              { ...lastMessage, content: lastMessage.content + token },
            ];
          }
          return prev;
        });
      }, language, showSpecialTokens);

      // 스트리밍이 끝난 후 마지막 assistant 응답 전체를 디버그용으로 브로드캐스트
      // assistantOutputRef에 누적된 텍스트를 사용 (메시지 상태는 비동기적으로 업데이트될 수 있음)
      setTimeout(() => {
        const finalText = assistantOutputRef.current || '';
        if (finalText) {
          // console.log('[ChatPage] Dispatching assistant-output event, text length:', finalText.length);
          window.dispatchEvent(new CustomEvent('assistant-output', {
            detail: { text: finalText },
          }));
        } else {
          // 메시지 상태에서 가져오기 시도
          setMessages(prev => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.content) {
              window.dispatchEvent(new CustomEvent('assistant-output', {
                detail: { text: lastMessage.content },
              }));
            }
            return prev;
          });
        }
      }, 100);
    } catch (error) {
      console.error('채팅 오류:', error);
      setIsWaitingForFirstToken(false);
      
      // 503 에러 (모델 로딩 중) 감지
      if (error.message && error.message.includes('503')) {
        setIsModelLoading(true);
        const errorMessage = { role: 'assistant', content: `모델이 로딩 중입니다. 잠시 후 다시 시도해주세요.` };
        setMessages(prev => [...prev.slice(0, -1), errorMessage]);
      } else {
        const errorMessage = { role: 'assistant', content: `Error: ${error.message}` };
        setMessages(prev => [...prev.slice(0, -1), errorMessage]);
      }
    } finally {
      setIsLoading(false);
      setIsWaitingForFirstToken(false);
      // 토큰 생성 완료 후 입력 필드로 포커스 이동
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    if (isLoading) return; // 로딩 중이면 초기화 불가
    setMessages([]);
    setInput('');
    
    // Context 사용량 리셋
    try {
      const config = JSON.parse(localStorage.getItem('modelConfig')) || {};
      const contextSize = config.contextSize || 2048;
      window.dispatchEvent(new CustomEvent('context-update', {
        detail: {
          used: 0,
          total: contextSize
        }
      }));
    } catch (error) {
      console.error('Failed to reset context usage:', error);
    }
  };

  return (
    <PanelGroup direction="horizontal" className="chat-page-layout">
      <Panel defaultSize={70} minSize={30}>
        <div className="chat-panel">
          <ProgressBar loading={isModelLoading} />
          <div className="messages-container">
            {messages.length === 0 && !isLoading ? (
              <div className="welcome-message">{t('chat.welcome')}</div>
            ) : (
              messages.map((msg, index) => {
                // 스페셜 토큰 하이라이트 함수
                const renderContent = (content) => {
                  // 스페셜 토큰 제거 (챗 버블에는 스페셜 토큰이 표시되면 안됨)
                  let cleanedContent = content.replace(/<\|[^>]*\|>/g, '');
                  
                  if (!showSpecialTokens) {
                    return cleanedContent;
                  }
                  // 스페셜 토큰 표시가 켜져있어도 챗 버블에는 표시하지 않음
                  return cleanedContent;
                };

                // assistant 메시지이고 content가 비어있고 첫 토큰을 기다리는 중이면 로딩 애니메이션 표시
                const showLoading = msg.role === 'assistant' && !msg.content && isWaitingForFirstToken && index === messages.length - 1;

                return (
                  <div key={index} className={`message ${msg.role}`}>
                    <div className="message-content">
                      {showLoading ? (
                        <div className="loading-dots">
                          <span>.</span><span>.</span><span>.</span>
                        </div>
                      ) : (
                        renderContent(msg.content)
                      )}
                    </div>
                  </div>
                );
              })
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chat-input-area">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={t('chat.placeholder')}
              rows="3"
              disabled={isLoading}
            />
            <div className="chat-buttons">
              <button onClick={handleSend} disabled={isLoading}>
                {t('chat.send')}
              </button>
              <button 
                onClick={handleClear} 
                disabled={isLoading}
                className="clear-button"
              >
                {t('chat.clear')}
              </button>
            </div>
          </div>
        </div>
      </Panel>
      <PanelResizeHandle className="resize-handle" />
      <Panel defaultSize={30} minSize={20}>
        <LogPanel />
      </Panel>
    </PanelGroup>
  );
};

export default ChatPage;
