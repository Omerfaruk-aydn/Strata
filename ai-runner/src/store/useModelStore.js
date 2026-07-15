/**
 * AI Runner — Model Store
 * Manages installed models, search results, download state, and active model.
 * Zustand store with REST API integration.
 */

import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8420';

const useModelStore = create((set, get) => ({
  // ── State ──
  localModels: [],
  searchResults: [],
  activeModel: null,
  searchQuery: '',
  isSearching: false,
  isLoading: false,
  downloadProgress: {},  // { [modelId]: { progress, status, speed, eta } }
  loadingModelId: null,
  error: null,

  // ── Actions ──

  /** FR-101: Search HuggingFace for GGUF models */
  searchModels: async (query) => {
    set({ isSearching: true, searchQuery: query, error: null });
    try {
      const res = await fetch(`${API_BASE}/api/models/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error('Arama başarısız');
      const data = await res.json();
      set({ searchResults: data.models || [], isSearching: false });
    } catch (err) {
      set({ error: err.message, isSearching: false });
    }
  },

  /** FR-105: Load local model library */
  fetchLocalModels: async () => {
    try {
      const res = await fetch(`${API_BASE}/api/models/local`);
      if (!res.ok) throw new Error('Yerel modeller yüklenemedi');
      const data = await res.json();
      set({ localModels: data.models || [] });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-102: Download a model with progress tracking */
  downloadModel: async (modelId, quant = 'Q4_K_M') => {
    set((state) => ({
      downloadProgress: {
        ...state.downloadProgress,
        [modelId]: { progress: 0, status: 'downloading', speed: 0, eta: 0 },
      },
    }));

    try {
      const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quant }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              set((state) => ({
                downloadProgress: {
                  ...state.downloadProgress,
                  [modelId]: {
                    progress: data.progress || 0,
                    status: data.status || 'downloading',
                    speed: data.speed_mbps || 0,
                    eta: data.eta_seconds || 0,
                    downloadedBytes: data.downloaded_bytes || 0,
                    totalBytes: data.total_bytes || 0,
                  },
                },
              }));

              if (data.status === 'completed') {
                // Refresh local models
                get().fetchLocalModels();
              }
            } catch (e) { /* skip invalid JSON */ }
          }
        }
      }
    } catch (err) {
      set((state) => ({
        downloadProgress: {
          ...state.downloadProgress,
          [modelId]: { progress: 0, status: 'error' },
        },
        error: err.message,
      }));
    }
  },

  /** Load model into memory for inference with all performance optimizations */
  loadModel: async (modelId, options = {}) => {
    set({ loadingModelId: modelId, error: null });
    try {
      const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          quant:               options.quant              || 'Q4_K_M',
          n_gpu_layers:        options.nGpuLayers         ?? null,
          context_length:      options.contextLength       || 4096,
          n_threads:           options.nThreads            ?? null,
          n_batch:             options.nBatch              || 512,
          // Memory
          use_mmap:            options.useMmap             ?? true,
          use_mlock:           options.useMlock            ?? true,
          // KV Cache quantization (default: q4_0 = 50% VRAM saving)
          kv_cache_type:       options.kvCacheType         || 'q4_0',
          // Flash Attention
          flash_attn:          options.flashAttn           ?? true,
          // Smart Context Shifting
          cache_context_shift: options.cacheContextShift   ?? true,
          // Speculative Decoding (optional)
          draft_model_path:    options.draftModelPath       || null,
          draft_n_gpu_layers:  options.draftNGpuLayers      ?? -1,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Model yüklenemedi');
      }

      const data = await res.json();
      set({
        activeModel:    data.model,
        optimizations:  data.optimizations || {},
        loadingModelId: null,
      });
      return data.model;
    } catch (err) {
      set({ error: err.message, loadingModelId: null });
      throw err;
    }
  },


  /** Unload active model */
  unloadModel: async () => {
    try {
      await fetch(`${API_BASE}/api/models/unload`, { method: 'POST' });
      set({ activeModel: null });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-104: Get offload plan preview */
  getOffloadPlan: async (modelId, quant = 'Q4_K_M', contextLength = 4096) => {
    try {
      const res = await fetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quant, context_length: contextLength }),
      });
      if (!res.ok) throw new Error('Plan hesaplanamadı');
      return await res.json();
    } catch (err) {
      set({ error: err.message });
      return null;
    }
  },

  /** FR-105: Delete a local model */
  deleteModel: async (modelId) => {
    try {
      const res = await fetch(`${API_BASE}/api/models/local/${encodeURIComponent(modelId)}`, {
        method: 'DELETE',
      });
      if (!res.ok) throw new Error('Model silinemedi');
      get().fetchLocalModels();
    } catch (err) {
      set({ error: err.message });
    }
  },

  clearError: () => set({ error: null }),
  clearSearch: () => set({ searchResults: [], searchQuery: '' }),
}));

export default useModelStore;
