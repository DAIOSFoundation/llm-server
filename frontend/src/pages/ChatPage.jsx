import React, { useState, useEffect, useRef } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { sendChatMessage } from '../services/api';
import ProgressBar from '../components/ProgressBar';
import ServerInfoPanel from '../components/ServerInfoPanel';
import './ChatPage.css';

const ChatPage = () => {
  const { t } = useLanguage();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === '' || isLoading) return;

    const newUserMessage = { role: 'user', content: input, timestamp: Date.now() };
    const currentMessages = [...messages, newUserMessage];
    setMessages(currentMessages);
    setInput('');
    setIsLoading(true);

    const assistantMessageId = Date.now() + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '', timestamp: assistantMessageId }]);
    
    try {
      // The new API handles streaming internally via the onToken callback
      await sendChatMessage(
        currentMessages,
        (token) => {
          if (isLoading) setIsLoading(false); // Hide progress bar on first token
          setMessages(prev =>
            prev.map(msg =>
              msg.timestamp === assistantMessageId
                ? { ...msg, content: msg.content + token }
                : msg
            )
          );
        }
      );
    } catch (error) {
      console.error('채팅 오류:', error);
       setMessages(prev =>
        prev.map(msg =>
          msg.timestamp === assistantMessageId
            ? { ...msg, content: '오류가 발생했습니다. 다시 시도해주세요.' }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  return (
    <div className="chat-page-container">
      <div className="chat-page">
        <ProgressBar loading={isLoading && messages[messages.length - 1]?.content === ''} />
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="chat-empty">
              <h1>BanyaLLM</h1>
              <p>{t('chat.welcome')}</p>
            </div>
          )}
          {messages.map((msg) => (
            <div key={msg.timestamp} className={`chat-bubble ${msg.role}`}>
              <p>{msg.content}</p>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
        <div className="chat-input-container">
          <input
            type="text"
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={t('chat.placeholder')}
            disabled={isLoading}
          />
          <button className="send-button" onClick={handleSend} disabled={isLoading}>
            {t('chat.send')}
          </button>
        </div>
      </div>
      <ServerInfoPanel />
    </div>
  );
};

export default ChatPage;

