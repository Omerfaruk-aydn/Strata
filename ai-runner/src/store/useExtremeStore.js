import { create } from 'zustand';
import { apiFetch } from '../api/client';

const useExtremeStore = create((set, get) => ({
  capabilities: null,
  presets: [],
  report: null,
  metadata: null,
  profiles: [],
  benchmark: null,
  quantization: { available: false, supported_quants: [], jobs: [] },
  ultraCapabilities: null,
  ultraMemoryReport: null,
  ultraBenchmark: null,
  ultraModels: [],
  isLoading: false,
  isBenchmarking: false,
  isQuantizing: false,
  error: null,

  fetchCapabilities: async (refresh = false) => {
    try {
      const res = await apiFetch(`/api/extreme/capabilities${refresh ? '?refresh=true' : ''}`);
      const data = await res.json();
      set({ capabilities: data });
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  fetchUltraCapabilities: async () => {
    try {
      const res = await apiFetch('/api/ultra/capabilities');
      const data = await res.json();
      set({ ultraCapabilities: data });
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  fetchUltraModels: async () => {
    try {
      const res = await apiFetch('/api/ultra/models');
      const data = await res.json();
      set({ ultraModels: data.models || [] });
      return data.models || [];
    } catch (error) {
      set({ error: error.message });
      return [];
    }
  },

  fetchStrataLayout: async (modelFile) => {
    try {
      const res = await apiFetch(`/api/ultra/layout/${encodeURIComponent(modelFile)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata layout discovery failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  estimateUltraMemory: async (valueCount = 4096, groupSize = 128) => {
    try {
      const res = await apiFetch('/api/ultra/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value_count: valueCount, group_size: groupSize }),
      });
      const data = await res.json();
      set({ ultraMemoryReport: data.report });
      return data.report;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  measureStrataQuality: async (reference, reconstructed) => {
    try {
      const res = await apiFetch('/api/ultra/quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference, reconstructed }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata quality measurement failed');
      return data.quality;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  runUltraBenchmark: async (valueCount = 16384, groupSize = 128) => {
    try {
      const res = await apiFetch('/api/ultra/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value_count: valueCount, group_size: groupSize }),
      });
      const data = await res.json();
      set({ ultraBenchmark: data.benchmark });
      return data.benchmark;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  convertToStrata: async (modelId, targetName = null, groupSize = 128, targetCodec = 'ternary-q05', sparseThreshold = 0.125) => {
    try {
      const res = await apiFetch(`/api/ultra/convert/${encodeURIComponent(modelId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_name: targetName, group_size: groupSize, target_codec: targetCodec, sparse_threshold: sparseThreshold }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata conversion failed');
      return data.conversion;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  runStrataMatvec: async (modelFile, tensorName, vector, options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/matvec', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: modelFile,
          tensor_name: tensorName,
          vector,
          memory_budget_bytes: options.memoryBudgetBytes || 512 * 1024 * 1024,
          resident_window: options.residentWindow || 2,
          backend: options.backend || 'auto',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata runtime execution failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  benchmarkStrataRuntime: async (modelFile, tensorName, vector, options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/runtime-benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: modelFile,
          tensor_name: tensorName,
          vector,
          iterations: options.iterations || 10,
          memory_budget_bytes: options.memoryBudgetBytes || 512 * 1024 * 1024,
          resident_window: options.residentWindow || 2,
          backend: options.backend || 'auto',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata runtime benchmark failed');
      return data.benchmark;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  runStrataGraph: async (modelFile, nodes, vector, options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/graph/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: modelFile,
          nodes,
          vector,
          memory_budget_bytes: options.memoryBudgetBytes || 512 * 1024 * 1024,
          resident_window: options.residentWindow || 2,
          backend: options.backend || 'auto',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata graph execution failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  runLowBitAttentionStep: async (width, query, key, value, options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/attention/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          width,
          query,
          key,
          value,
          mode: options.mode || 'sign1',
          capacity_tokens: options.capacityTokens || 2048,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Low-bit attention failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  runStrataTransformerStep: async (modelFile, blockPrefixes, width, hidden, options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/transformer/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: modelFile,
          block_prefixes: blockPrefixes,
          width,
          hidden,
          context_capacity: options.contextCapacity || 2048,
          kv_mode: options.kvMode || 'sign1',
          memory_budget_bytes: options.memoryBudgetBytes || 512 * 1024 * 1024,
          resident_window: options.residentWindow || 2,
          backend: options.backend || 'auto',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata transformer execution failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  generateStrataText: async (options = {}) => {
    try {
      const res = await apiFetch('/api/ultra/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: options.modelFile,
          block_prefixes: options.blockPrefixes || [],
          embedding_tensor: options.embeddingTensor,
          output_tensor: options.outputTensor,
          width: options.width,
          prompt: options.prompt || '',
          max_new_tokens: options.maxNewTokens || 16,
          context_capacity: options.contextCapacity || 2048,
          kv_mode: options.kvMode || 'sign1',
          memory_budget_bytes: options.memoryBudgetBytes || 512 * 1024 * 1024,
          resident_window: options.residentWindow || 2,
          backend: options.backend || 'auto',
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Strata generation failed');
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  fetchPresets: async () => {
    try {
      const res = await apiFetch('/api/extreme/presets');
      const data = await res.json();
      set({ presets: data.presets || [] });
    } catch (error) {
      set({ error: error.message });
    }
  },

  analyzeLocal: async (modelId, options = {}) => {
    set({ isLoading: true, error: null, benchmark: null });
    try {
      const res = await apiFetch(`/api/extreme/analyze/${encodeURIComponent(modelId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          quant: options.quant || null,
          preset: options.preset || 'maximum_capacity',
          context_length: options.contextLength || 2048,
          selected_gpu_index: options.selectedGpuIndex ?? 0,
          tensor_split: options.tensorSplit || null,
        }),
      });
      const data = await res.json();
      set({ report: data.report, metadata: data.metadata || null, isLoading: false });
      return data.report;
    } catch (error) {
      set({ error: error.message, report: null, isLoading: false });
      return null;
    }
  },

  simulate: async (options) => {
    set({ isLoading: true, error: null, benchmark: null });
    try {
      const res = await apiFetch('/api/extreme/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: options.modelId || `simulation-${options.parameterB}B`,
          parameter_count: Math.round(Number(options.parameterB) * 1_000_000_000),
          quant: options.quant || 'Q4_K_M',
          preset: options.preset || 'maximum_capacity',
          context_length: options.contextLength || 2048,
          native_context_length: options.nativeContextLength || 4096,
          selected_gpu_index: options.selectedGpuIndex ?? 0,
        }),
      });
      const data = await res.json();
      set({ report: data.report, metadata: null, isLoading: false });
      return data.report;
    } catch (error) {
      set({ error: error.message, report: null, isLoading: false });
      return null;
    }
  },

  runBenchmark: async (maxTokens = 32) => {
    set({ isBenchmarking: true, error: null });
    try {
      const res = await apiFetch('/api/extreme/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_tokens: maxTokens }),
      });
      const data = await res.json();
      set({ benchmark: data.benchmark, isBenchmarking: false });
      await get().fetchProfiles();
      return data.benchmark;
    } catch (error) {
      set({ error: error.message, isBenchmarking: false });
      return null;
    }
  },

  rebalance: async (preset = 'maximum_capacity') => {
    set({ isLoading: true, error: null });
    try {
      const res = await apiFetch('/api/extreme/rebalance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset }),
      });
      const data = await res.json();
      set({ report: data.report || get().report, isLoading: false });
      return data;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      return null;
    }
  },

  fetchProfiles: async (modelId = null) => {
    try {
      const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
      const res = await apiFetch(`/api/extreme/profiles${query}`);
      const data = await res.json();
      set({ profiles: data.profiles || [] });
    } catch (error) {
      set({ error: error.message });
    }
  },

  fetchQuantization: async () => {
    try {
      const res = await apiFetch('/api/extreme/quantization');
      const data = await res.json();
      set({ quantization: data });
      return data;
    } catch (error) {
      set({ error: error.message });
      return null;
    }
  },

  startQuantization: async (modelId, sourceQuant, outputQuant, allowRequantize = false) => {
    set({ isQuantizing: true, error: null });
    try {
      const res = await apiFetch(`/api/extreme/quantization/start/${encodeURIComponent(modelId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          quant: outputQuant,
          source_quant: sourceQuant || null,
          allow_requantize: allowRequantize,
        }),
      });
      const data = await res.json();
      await get().fetchQuantization();
      set({ isQuantizing: false });
      return data.job;
    } catch (error) {
      set({ error: error.message, isQuantizing: false });
      return null;
    }
  },

  cancelQuantization: async (jobId) => {
    try {
      await apiFetch(`/api/extreme/quantization/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
      await get().fetchQuantization();
    } catch (error) {
      set({ error: error.message });
    }
  },

  clearError: () => set({ error: null }),
  clearReport: () => set({ report: null, metadata: null, benchmark: null }),
}));

export default useExtremeStore;
