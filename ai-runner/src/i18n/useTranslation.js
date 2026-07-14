/**
 * AI Runner — i18n Hook
 * Simple translation hook using JSON locale files.
 */

import { useMemo } from 'react';
import useSettingsStore from '../store/useSettingsStore';
import tr from './tr.json';
import en from './en.json';

const locales = { tr, en };

/**
 * Translation hook. Returns a `t` function that resolves dot-notation keys.
 * Usage: const t = useTranslation(); t('models.download')
 */
export function useTranslation() {
  const language = useSettingsStore((s) => s.language);

  const t = useMemo(() => {
    const locale = locales[language] || locales.tr;

    return (key, replacements = {}) => {
      const keys = key.split('.');
      let value = locale;

      for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
          value = value[k];
        } else {
          return key; // Fallback to key
        }
      }

      if (typeof value !== 'string') return key;

      // Simple template replacement: {{var}}
      return value.replace(/\{\{(\w+)\}\}/g, (_, k) => replacements[k] ?? `{{${k}}}`);
    };
  }, [language]);

  return t;
}

export default useTranslation;
