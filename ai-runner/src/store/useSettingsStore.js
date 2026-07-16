/**
 * AI Runner — Settings Store
 * User preferences, theme, language.
 * Implements FR-601–FR-604, FR-407.
 */

import { create } from 'zustand';
import {
  apiFetch,
  configureApi,
  getApiConfig,
  scheduleApiEndpoint,
} from '../api/client';

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
  apiKeySource: 'disabled',
  allowNetworkAccess: false,
  advancedMode: false,
  isLoaded: false,
  isSaving: false,
  restartRequired: false,
  error: null,

  // ── Performance Optimization Settings ──
  kvCacheType: 'q4_0',        // KV Cache quantization: q4_0 | q5_0 | q8_0 | f16
  flashAttn: true,            // Backend-dependent acceleration on long contexts
  useMlock: true,             // Lock model in RAM — prevents OS swap
  cacheContextShift: true,    // Application-level context pruning toggle
  speculativeDecoding: false,
  draftNumPredTokens: 10,

  // ── Prompt Pruning (Context Management) ──
  maxContextLength: 4096,     // Max context window in tokens
  maxHistoryMessages: 20,     // Keep last N messages (0 = unlimited)
  autoContextPrune: true,     // Trim old messages when they exceed the prompt budget
  selectedGpuIndex: 0,
  tensorSplit: null,
  contextCompactionMode: 'extractive_summary',

  // ── Extreme Model Mode ──
  extremeModeEnabled: true,
  extremePreset: 'maximum_capacity',
  adaptiveLoad: true,
  adaptiveMaxAttempts: 6,
  backendPreference: 'auto',

  // ── Actions ──

  /** Load settings from backend */
  fetchSettings: async () => {
    try {
      const res = await apiFetch('/api/settings');
      const data = await res.json();
      const s = data.settings || {};
      const apiKeySource = data.api_key_source || (s.api_key ? 'settings' : 'disabled');

      // The API key takes effect immediately. Host/port are applied only on a
      // new app session because the running backend still listens at the old endpoint.
      if (apiKeySource === 'disabled') {
        configureApi({ apiKey: s.api_key });
      } else if (s.api_key) {
        // Older backends returned the key. Newer versions redact it, so keep
        // the key already entered in the startup recovery screen/local store.
        configureApi({ apiKey: s.api_key });
      }
      const effectiveApiKey = apiKeySource === 'disabled' ? null : getApiConfig().apiKey;

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
        apiKey: effectiveApiKey,
        apiKeySource,
        allowNetworkAccess: s.allow_network_access ?? false,
        advancedMode: s.advanced_mode || false,
        // Performance
        kvCacheType: s.kv_cache_type || 'q4_0',
        flashAttn: s.flash_attn ?? true,
        useMlock: s.use_mlock ?? true,
        cacheContextShift: s.cache_context_shift ?? true,
        speculativeDecoding: s.speculative_decoding ?? false,
        draftNumPredTokens: s.draft_num_pred_tokens ?? 10,
        // Prompt Pruning
        maxContextLength: s.max_context_length || 4096,
        maxHistoryMessages: s.max_history_messages ?? 20,
        autoContextPrune: s.auto_context_prune ?? true,
        selectedGpuIndex: s.selected_gpu_index ?? 0,
        tensorSplit: s.tensor_split || null,
        contextCompactionMode: s.context_compaction_mode || 'extractive_summary',
        extremeModeEnabled: s.extreme_mode_enabled ?? true,
        extremePreset: s.extreme_preset || 'maximum_capacity',
        adaptiveLoad: s.adaptive_load ?? true,
        adaptiveMaxAttempts: s.adaptive_max_attempts || 6,
        backendPreference: s.backend_preference || 'auto',
        isLoaded: true,
      });

      // Apply theme
      document.documentElement.setAttribute('data-theme', s.theme || 'dark');
      return true;

    } catch (err) {
      console.error('Settings fetch error:', err);
      set({ error: err.message, isLoaded: true });
      return false;
    }
  },

  /** Save settings to backend */
  saveSettings: async (updates) => {
    set({ isSaving: true, error: null });
    const current = get();
    const endpointChanged = (
      ('apiHost' in updates && updates.apiHost !== current.apiHost)
      || ('apiPort' in updates && Number(updates.apiPort) !== Number(current.apiPort))
    );
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
      allowNetworkAccess: 'allow_network_access',
      advancedMode: 'advanced_mode',
      // Performance
      kvCacheType: 'kv_cache_type',
      flashAttn: 'flash_attn',
      useMlock: 'use_mlock',
      cacheContextShift: 'cache_context_shift',
      speculativeDecoding: 'speculative_decoding',
      draftNumPredTokens: 'draft_num_pred_tokens',
      // Prompt Pruning
      maxContextLength: 'max_context_length',
      maxHistoryMessages: 'max_history_messages',
      autoContextPrune: 'auto_context_prune',
      selectedGpuIndex: 'selected_gpu_index',
      tensorSplit: 'tensor_split',
      contextCompactionMode: 'context_compaction_mode',
      extremeModeEnabled: 'extreme_mode_enabled',
      extremePreset: 'extreme_preset',
      adaptiveLoad: 'adaptive_load',
      adaptiveMaxAttempts: 'adaptive_max_attempts',
      backendPreference: 'backend_preference',
    };

    for (const [key, value] of Object.entries(updates)) {
      if (key === 'apiKey' && current.apiKeySource === 'environment') continue;
      const backendKey = keyMap[key] || key;
      backendSettings[backendKey] = value;
    }

    try {
      await apiFetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: backendSettings }),
      });

      set(updates);

      if ('apiKey' in updates && current.apiKeySource !== 'environment') {
        // Authentication changes on the current server as soon as it accepts the save.
        configureApi({ apiKey: updates.apiKey });
        set({ apiKeySource: updates.apiKey ? 'settings' : 'disabled' });
      }
      if (endpointChanged) {
        scheduleApiEndpoint({
          host: updates.apiHost ?? current.apiHost,
          port: updates.apiPort ?? current.apiPort,
        });
      }

      // Apply theme change immediately
      if (updates.theme) {
        document.documentElement.setAttribute('data-theme', updates.theme);
      }
      set({ isSaving: false, restartRequired: endpointChanged || current.restartRequired });
      return true;
    } catch (err) {
      console.error('Settings save error:', err);
      set({ isSaving: false, error: err.message });
      return false;
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
      const res = await apiFetch('/api/settings/export');
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
      await apiFetch('/api/settings/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings }),
      });
      if (Object.hasOwn(settings, 'api_key') && get().apiKeySource !== 'environment') {
        configureApi({ apiKey: settings.api_key });
      }
      if (Object.hasOwn(settings, 'api_host') || Object.hasOwn(settings, 'api_port')) {
        scheduleApiEndpoint({
          host: settings.api_host ?? get().apiHost,
          port: settings.api_port ?? get().apiPort,
        });
        set({ restartRequired: true });
      }
      await get().fetchSettings();
    } catch (err) {
      console.error('Import error:', err);
      set({ error: err.message });
    }
  },

  clearError: () => set({ error: null }),
}));

export default useSettingsStore;
