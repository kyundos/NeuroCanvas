import { useCallback, useMemo, useState, type ReactNode } from "react";
import { TranslationContext } from "./translationContext";
import {
  translations,
  type Language,
  type TranslationKey,
  type TranslateParams,
  LANGUAGE_STORAGE_KEY,
  replaceTemplateValues,
} from "./translations";

const getInitialLanguage = (): Language => {
  if (typeof window === "undefined") return "en";
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return saved === "it" ? "it" : "en";
};

export function TranslationProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(getInitialLanguage);

  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
    }
  }, []);

  const toggleLanguage = useCallback(() => {
    setLanguage(language === "en" ? "it" : "en");
  }, [language, setLanguage]);

  const t = useCallback(
    (key: TranslationKey, params?: TranslateParams) => {
      const template = translations[language][key];
      return replaceTemplateValues(template, params);
    },
    [language],
  );

  const value = useMemo(
    () => ({ language, setLanguage, toggleLanguage, t }),
    [language, setLanguage, toggleLanguage, t],
  );

  return (
    <TranslationContext.Provider value={value}>
      {children}
    </TranslationContext.Provider>
  );
}
