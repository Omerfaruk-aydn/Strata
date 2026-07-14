/**
 * AI Runner — Settings Store
 * User preferences, theme, language.
 * Implements FR-601–FR-604, FR-407.
 */

import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8420';

const useSettingsStore = create((set, get) => ({
  // ── State ──
  theme: 'dark',
  language: 'tr',
  defaultModel: null,
  defaultSystemPrompt: '',
  modelDir: '',
  cacheSizeLimitGb: 50,
  nThreads: null,
  useMmap: true,
  nBatch: 512,
  apiHost: '127.0.0.1',
  apiPort: 8420,
  apiKey: null,
  advancedMode: false,
  isLoaded: false,

  // ── Actions ──

  /** Load settings from backend */
  fetchSettings: async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`);
      if (!res.ok) return;
      const data = await res.json();
      const s = data.settings || {};

      set({
        theme: s.theme || 'dark',
        language: s.language || 'tr',
        defaultModel: s.default_model || null,
        defaultSystemPrompt: s.default_system_prompt || '',
        modelDir: s.model_dir || '',
        cacheSizeLimitGb: s.cache_size_limit_gb || 50,
        nThreads: s.n_threads || null,
        useMmap: s.use_mmap ?? true,
        nBatch: s.n_batch || 512,
        apiHost: s.api_host || '127.0.0.1',
        apiPort: s.api_port || 8420,
        apiKey: s.api_key || null,
        advancedMode: s.advanced_mode || false,
        isLoaded: true,
      });

      // Apply theme
      document.documentElement.setAttribute('data-theme', s.theme || 'dark');

    } catch (err) {
      console.error('Settings fetch error:', err);
    }
  },

  /** Save settings to backend */
  saveSettings: async (updates) => {
    // Map camelCase to snake_case for backend
    const backendSettings = {};
    const keyMap = {
      theme: 'theme',
      language: 'language',
      defaultModel: 'default_model',
      defaultSystemPrompt: 'default_system_prompt',
      modelDir: 'model_dir',
      cacheSizeLimitGb: 'cache_size_limit_gb',
      nThreads: 'n_threads',
      useMmap: 'use_mmap',
      nBatch: 'n_batch',
      apiHost: 'api_host',
      apiPort: 'api_port',
      apiKey: 'api_key',
      advancedMode: 'advanced_mode',
    };

    for (const [key, value] of Object.entries(updates)) {
      const backendKey = keyMap[key] || key;
      backendSettings[backendKey] = value;
    }

    try {
      await fetch(`${API_BASE}/api/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: backendSettings }),
      });

      set(updates);

      // Apply theme change immediately
      if (updates.theme) {
        document.documentElement.setAttribute('data-theme', updates.theme);
      }
    } catch (err) {
      console.error('Settings save error:', err);
    }
  },

  /** FR-407: Toggle dark/light theme */
  toggleTheme: () => {
    const newTheme = get().theme === 'dark' ? 'light' : 'dark';
    get().saveSettings({ theme: newTheme });
  },

  /** Toggle language (TR/EN) */
  toggleLanguage: () => {
    const newLang = get().language === 'tr' ? 'en' : 'tr';
    get().saveSettings({ language: newLang });
  },

  /** Toggle advanced mode */
  toggleAdvancedMode: () => {
    const newMode = !get().advancedMode;
    get().saveSettings({ advancedMode: newMode });
  },

  /** FR-604: Export all settings as JSON */
  exportSettings: async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings/export`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ai-runner-settings.json';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export error:', err);
    }
  },

  /** FR-604: Import settings from JSON file */
  importSettings: async (file) => {
    try {
      const text = await file.text();
      const settings = JSON.parse(text);
      await fetch(`${API_BASE}/api/settings/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings }),
      });
      get().fetchSettings(); // Reload
    } catch (err) {
      console.error('Import error:', err);
    }
  },
}));

export default useSettingsStore;
