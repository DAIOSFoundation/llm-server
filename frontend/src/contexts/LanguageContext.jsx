import React, { createContext, useState, useContext, useEffect } from 'react';
import rawTranslations from '../translations.json';

const LanguageContext = createContext();

const translationsData = rawTranslations.default || rawTranslations;

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(localStorage.getItem('language') || 'ko');
  
  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  const t = (key) => {
    const langTranslations = translationsData[language];
    if (langTranslations && typeof langTranslations[key] === 'string') {
      return langTranslations[key];
    }
    // Fallback to the key if the translation is not found
    return key;
  };
  
  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext);
