import { useEffect, useMemo, useState } from "react";

export type Lang = "zh" | "en";

const LANG_KEY = "oneEval.lang";
const LANG_EVENT = "oneEval:lang-change";

const detectLang = (): Lang => {
  const stored = localStorage.getItem(LANG_KEY);
  if (stored === "zh" || stored === "en") return stored;
  const browser = navigator.language.toLowerCase();
  return browser.startsWith("zh") ? "zh" : "en";
};

export const useLang = () => {
  const [lang, setLangState] = useState<Lang>(() => detectLang());

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key !== LANG_KEY) return;
      if (e.newValue === "zh" || e.newValue === "en") {
        setLangState(e.newValue);
      }
    };
    const onCustom = () => {
      const next = localStorage.getItem(LANG_KEY);
      if (next === "zh" || next === "en") setLangState(next);
    };
    window.addEventListener("storage", onStorage);
    window.addEventListener(LANG_EVENT, onCustom);
    return () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener(LANG_EVENT, onCustom);
    };
  }, []);

  const setLang = (next: Lang) => {
    localStorage.setItem(LANG_KEY, next);
    setLangState(next);
    window.dispatchEvent(new CustomEvent(LANG_EVENT));
  };

  const t = useMemo(
    () => (dict: Record<Lang, string>) => dict[lang] ?? dict.en,
    [lang]
  );

  return { lang, setLang, t };
};
