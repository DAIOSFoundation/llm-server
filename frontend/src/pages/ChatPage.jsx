import React, { useState, useEffect, useRef } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { sendChatMessage, buildLlama3Prompt, countTokens, LLAMA_BASE_URL } from '../services/api';
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

  // 모델 로딩 상태 체크
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        // 라우터 모드 포함: /health 로 서버 기동 여부만 체크
        const response = await fetch(`${LLAMA_BASE_URL}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(1000),
        });
        setIsModelLoading(false);
      } catch (error) {
        // 서버가 아직 시작되지 않았거나 연결 불가
        // 타임아웃이나 네트워크 에러는 조용히 처리 (로그 출력 안 함)
        if (error.name !== 'AbortError' && !error.message.includes('Failed to fetch')) {
          // 다른 에러만 로깅
        }
        // 에러가 발생해도 모델이 로딩 중일 수 있으므로 true로 설정하지 않음
        // (이미 다른 곳에서 설정되었을 수 있음)
      }
    };

    // 모델 로딩 이벤트 리스너
    const handleModelLoading = (event) => {
      setIsModelLoading(event.detail.loading);
    };

    window.addEventListener('model-loading', handleModelLoading);

    // 초기 체크
    checkServerStatus();

    // 주기적으로 체크 (1초마다)
    const interval = setInterval(checkServerStatus, 1000);

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
      window.dispatchEvent(new CustomEvent('prompt-output', {
        detail: { prompt: fullPrompt },
      }));
    } catch (e) {
      console.error('Failed to build prompt for debug:', e);
    }

    // 새 응답 시작 시 이전 assistant 출력 및 버퍼 초기화
    assistantOutputRef.current = '';
    streamBufferRef.current = '';

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantMessage = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, assistantMessage]);

    try {
      await sendChatMessage([...messages, userMessage], (token) => {
        // 토큰이 수신되면 모델 로딩 완료
        setIsModelLoading(false);
        // 디버깅용으로 전체 assistant 출력 누적
        assistantOutputRef.current += token;
        
        // 지연된 스트리밍: 토큰을 버퍼에 쌓고, 문장 종결 부호가 있을 때만 화면에 반영
        streamBufferRef.current += token;
        const sentenceEndings = ['.', '!', '?', '\n']; // 줄바꿈도 문장 끝으로 간주
        
        // 버퍼에 종결 부호가 있는지 확인
        let lastEndingIndex = -1;
        for (const ending of sentenceEndings) {
          const index = streamBufferRef.current.lastIndexOf(ending);
          if (index > lastEndingIndex) {
            lastEndingIndex = index;
          }
        }
        
        // 종결 부호가 있으면 그 지점까지 잘라서 화면에 업데이트
        if (lastEndingIndex !== -1) {
          const chunkToRender = streamBufferRef.current.substring(0, lastEndingIndex + 1);
          streamBufferRef.current = streamBufferRef.current.substring(lastEndingIndex + 1); // 남은 부분은 버퍼에 유지
          
          setMessages(prev => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage && lastMessage.role === 'assistant') {
              // 스페셜 토큰 표시 설정에 따라 처리 (이전 로직 활용)
              const currentContent = lastMessage.content + chunkToRender;
              return [
                ...prev.slice(0, -1),
                { ...lastMessage, content: currentContent },
              ];
            }
            return prev;
          });
        }
      }, language, showSpecialTokens);

      // 스트리밍이 끝난 후 마지막 assistant 응답 전체를 디버그용으로 브로드캐스트
      // (화면에는 이미 '완성된 문장'까지만 보여주고 있고, 버퍼에 남은 찌꺼기는 버림으로써 '사라지는 현상' 방지)
      if (assistantOutputRef.current) {
        // 실제 화면에 표시된 텍스트와 디버그용 텍스트는 다를 수 있음 (버퍼 잔여물 때문)
        // 디버그용으로는 전체를 다 보내줌
        window.dispatchEvent(new CustomEvent('assistant-output', {
          detail: { text: assistantOutputRef.current },
        }));
      }
    } catch (error) {
      console.error('채팅 오류:', error);
      
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
                  if (!showSpecialTokens) {
                    return content;
                  }
                  // 스페셜 토큰 패턴 찾기: <|...|> (|>로 끝나는 모든 패턴)
                  const parts = content.split(/(<\|[^>]*\|>)/g);
                  return parts.map((part, i) => {
                    if (part.match(/^<\|[^>]*\|>$/)) {
                      return <span key={i} className="special-token">{part}</span>;
                    }
                    return part;
                  });
                };

                return (
                  <div key={index} className={`message ${msg.role}`}>
                    <div className="message-content">{renderContent(msg.content)}</div>
                  </div>
                );
              })
            )}
            {isLoading && messages[messages.length - 1]?.role === 'user' && (
              <div className="message assistant">
                <div className="message-content loading-dots">
                  <span>.</span><span>.</span><span>.</span>
                </div>
              </div>
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
