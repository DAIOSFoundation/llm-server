import React, { createContext, useContext, useState, useEffect } from 'react';

const LanguageContext = createContext();

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within LanguageProvider');
  }
  return context;
};

const translations = {
  ko: {
    chat: {
      title: '챗팅',
      placeholder: '메시지를 입력하세요...',
      send: '전송',
      selectModel: '모델 선택',
      noModel: '모델이 없습니다',
      loading: '생성 중...'
    },
    settings: {
      title: '설정',
      models: '모델 관리',
      addModel: '모델 추가',
      modelName: '모델 이름',
      modelPath: '모델 경로',
      selectFile: '파일 선택',
      quantization: '양자화 방법',
      device: '디바이스',
      gpuBackend: 'GPU 백엔드',
      inference: '추론 설정',
      contextSize: '컨텍스트 크기',
      batchSize: '배치 크기',
      maxTokens: '최대 토큰 수',
      temperature: 'Temperature',
      topK: 'Top K',
      topP: 'Top P',
      repeatPenalty: '반복 패널티',
      repeatLastN: '반복 마지막 N',
      threads: '스레드 수',
      threadsBatch: '배치 스레드 수',
      seed: '시드 (-1: 랜덤)',
      stopSequences: '중지 시퀀스',
      save: '저장',
      cancel: '취소',
      delete: '삭제',
      edit: '편집'
    },
    header: {
      settings: '설정',
      login: '로그인',
      language: '언어'
    }
  },
  en: {
    chat: {
      title: 'Chat',
      placeholder: 'Type a message...',
      send: 'Send',
      selectModel: 'Select Model',
      noModel: 'No models available',
      loading: 'Generating...'
    },
    settings: {
      title: 'Settings',
      models: 'Model Management',
      addModel: 'Add Model',
      modelName: 'Model Name',
      modelPath: 'Model Path',
      selectFile: 'Select File',
      quantization: 'Quantization',
      device: 'Device',
      gpuBackend: 'GPU Backend',
      inference: 'Inference Settings',
      contextSize: 'Context Size',
      batchSize: 'Batch Size',
      maxTokens: 'Max Tokens',
      temperature: 'Temperature',
      topK: 'Top K',
      topP: 'Top P',
      repeatPenalty: 'Repeat Penalty',
      repeatLastN: 'Repeat Last N',
      threads: 'Threads',
      threadsBatch: 'Batch Threads',
      seed: 'Seed (-1: random)',
      stopSequences: 'Stop Sequences',
      save: 'Save',
      cancel: 'Cancel',
      delete: 'Delete',
      edit: 'Edit'
    },
    header: {
      settings: 'Settings',
      login: 'Login',
      language: 'Language'
    }
  }
};

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(() => {
    const saved = localStorage.getItem('language');
    return saved || 'ko';
  });

  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  const t = (key) => {
    const keys = key.split('.');
    let value = translations[language];
    for (const k of keys) {
      value = value?.[k];
    }
    return value || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

