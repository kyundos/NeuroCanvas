import { createContext, useContext } from "react";
import type { Language, TranslationKey, TranslateParams } from "./translations";

export interface TranslationContextValue {
  language: Language;
  setLanguage: (next: Language) => void;
  toggleLanguage: () => void;
  t: (key: TranslationKey, params?: TranslateParams) => string;
}

export const TranslationContext = createContext<TranslationContextValue | null>(
  null,
);

export function useTranslation() {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error("useTranslation must be used within TranslationProvider.");
  }
  return context;
}
