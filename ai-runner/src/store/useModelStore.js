/**
 * AI Runner — Model Store
 * Manages installed models, search results, download state, and active model.
 * Zustand store with REST API integration.
 */

import { create } from 'zustand';
import { apiFetch, readSse } from '../api/client';

let searchController = null;
let searchRequestId = 0;

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
  loadReport: null,
  feasibility: null,
  runtimeProfile: null,
  offloadPlan: null,

  // ── Actions ──

  /** FR-101: Search HuggingFace for GGUF models */
  searchModels: async (query) => {
    searchController?.abort();
    searchController = new AbortController();
    const requestId = ++searchRequestId;
    set({ isSearching: true, searchQuery: query, error: null });
    try {
      const res = await apiFetch(`/api/models/search?q=${encodeURIComponent(query)}`, {
        signal: searchController.signal,
      });
      const data = await res.json();
      if (requestId === searchRequestId) {
        set({ searchResults: data.models || [], isSearching: false });
      }
    } catch (err) {
      if (err.name !== 'AbortError' && requestId === searchRequestId) {
        set({ error: err.message, isSearching: false });
      }
    } finally {
      if (requestId === searchRequestId) searchController = null;
    }
  },

  /** FR-105: Load local model library */
  fetchLocalModels: async () => {
    try {
      const res = await apiFetch('/api/models/local');
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
          [modelId]: { quant, progress: 0, status: 'downloading', speed: 0, eta: 0 },
      },
    }));

    let terminalStatus = null;
    try {
      const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quant }),
      });

      for await (const data of readSse(res)) {
        if (data.status === 'error') {
          terminalStatus = 'error';
          throw new Error(data.error || 'Model indirilemedi.');
        }

        set((state) => ({
          downloadProgress: {
            ...state.downloadProgress,
            [modelId]: (() => {
              const previous = state.downloadProgress[modelId] || {};
              return {
                quant: data.quant || quant,
                progress: data.status === 'completed' ? 1 : (data.progress ?? previous.progress ?? 0),
                status: data.status || 'downloading',
                speed: data.speed_mbps ?? previous.speed ?? 0,
                eta: data.eta_seconds ?? previous.eta ?? 0,
                downloadedBytes: data.downloaded_bytes ?? previous.downloadedBytes ?? 0,
                totalBytes: data.total_bytes ?? previous.totalBytes ?? 0,
              };
            })(),
          },
        }));

        if (['completed', 'paused', 'error'].includes(data.status)) {
          terminalStatus = data.status;
        }
        if (data.status === 'completed') await get().fetchLocalModels();
      }
      if (!terminalStatus) throw new Error('İndirme akışı tamamlanmadan kapandı.');
    } catch (err) {
      set((state) => ({
        downloadProgress: {
          ...state.downloadProgress,
          [modelId]: { quant, progress: 0, status: 'error' },
        },
        error: err.message,
      }));
    }
  },

  /** Pause an active download; its .part file is retained for resume. */
  cancelDownload: async (modelId) => {
    try {
      set((state) => ({
        downloadProgress: {
          ...state.downloadProgress,
          [modelId]: { ...state.downloadProgress[modelId], status: 'cancelling' },
        },
      }));
      await apiFetch(`/api/models/${encodeURIComponent(modelId)}/download/cancel`, {
        method: 'POST',
      });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** Load model into memory for inference with all performance optimizations */
  loadModel: async (modelId, options = {}) => {
    set({ loadingModelId: modelId, error: null });
    try {
      const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/load`, {
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
          speculative_decoding: options.speculativeDecoding ?? false,
          draft_num_pred_tokens: options.draftNumPredTokens ?? 10,
          selected_gpu_index:  options.selectedGpuIndex     ?? 0,
          tensor_split:        options.tensorSplit          || null,
          context_compaction_mode: options.contextCompactionMode || 'extractive_summary',
          extreme_preset:      options.extremePreset         || 'maximum_capacity',
          adaptive_load:       options.adaptiveLoad          ?? true,
          adaptive_max_attempts: options.adaptiveMaxAttempts || 6,
          backend_preference:  options.backendPreference     || 'auto',
        }),
      });

      const data = await res.json();
      set({
        activeModel:    data.model,
        optimizations:  data.optimizations || {},
        loadingModelId: null,
        loadReport:     data.load_report || null,
        feasibility:    data.feasibility || null,
        runtimeProfile: data.runtime_profile || null,
      });
      return data.model;
    } catch (err) {
      set({
        error: err.message,
        loadingModelId: null,
        loadReport: err.data?.detail?.load_report || null,
      });
      throw err;
    }
  },


  /** Unload active model */
  unloadModel: async () => {
    try {
      await apiFetch('/api/models/unload', { method: 'POST' });
      set({ activeModel: null, loadReport: null, feasibility: null, runtimeProfile: null, offloadPlan: null });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-104: Get offload plan preview */
  getOffloadPlan: async (modelId, quant = 'Q4_K_M', contextLength = 4096) => {
    try {
      const res = await apiFetch(`/api/models/${encodeURIComponent(modelId)}/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ quant, context_length: contextLength }),
      });
      const data = await res.json();
      set({ offloadPlan: data });
      return data;
    } catch (err) {
      set({ error: err.message });
      return null;
    }
  },

  /** FR-105: Delete a local model */
  deleteModel: async (modelId, quant = null) => {
    try {
      const query = quant ? `?quant=${encodeURIComponent(quant)}` : '';
      await apiFetch(`/api/models/local/${encodeURIComponent(modelId)}${query}`, {
        method: 'DELETE',
      });
      get().fetchLocalModels();
    } catch (err) {
      set({ error: err.message });
    }
  },

  clearError: () => set({ error: null }),
  clearSearch: () => {
    searchController?.abort();
    searchController = null;
    searchRequestId += 1;
    set({ searchResults: [], searchQuery: '', isSearching: false });
  },
}));

export default useModelStore;
