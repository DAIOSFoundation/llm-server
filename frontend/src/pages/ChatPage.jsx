import React, { useState, useEffect, useRef } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { sendChatMessage } from '../services/api';
import './ChatPage.css';

const ChatPage = () => {
  const { t } = useLanguage();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const selectedModelId = localStorage.getItem('selectedModelId');

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !selectedModelId || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const chatMessages = [...messages, userMessage].map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      const response = await sendChatMessage(selectedModelId, chatMessages, false);
      
      if (response.success) {
        const assistantMessage = {
          role: 'assistant',
          content: response.content,
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error(response.error || '응답 생성 실패');
      }
    } catch (error) {
      console.error('채팅 오류:', error);
      const errorMessage = {
        role: 'assistant',
        content: `오류: ${error.message}`,
        timestamp: Date.now()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-page">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>{t('chat.placeholder')}</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${message.role === 'user' ? 'user' : 'assistant'}`}
          >
            <div className="chat-message-content">
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="chat-message assistant">
            <div className="chat-message-content">
              {t('chat.loading')}
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-container">
        <textarea
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={t('chat.placeholder')}
          disabled={isLoading || !selectedModelId}
          rows={1}
        />
        <button
          className="chat-send-button"
          onClick={handleSend}
          disabled={!input.trim() || isLoading || !selectedModelId}
        >
          {t('chat.send')}
        </button>
      </div>
    </div>
  );
};

export default ChatPage;

