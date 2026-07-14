/**
 * AI Runner — Telemetry Store
 * WebSocket singleton for real-time telemetry data.
 * Implements FR-402, FR-403 live feeds.
 */

import { create } from 'zustand';

const WS_URL = 'ws://127.0.0.1:8420/ws/inference';

const useTelemetryStore = create((set, get) => ({
  // ── State ──
  connected: false,
  vramUsedMb: 0,
  vramTotalMb: 0,
  vramFreeMb: 0,
  ramUsedMb: 0,
  ramTotalMb: 0,
  ramFreeMb: 0,
  ramPercent: 0,
  gpuTemperature: null,
  tokensPerSec: 0,
  ttftMs: 0,
  modelLoaded: false,
  modelId: null,
  isGenerating: false,
  nGpuLayers: 0,
  contextLength: 0,

  // Layer distribution (FR-403)
  layerDistribution: {
    gpu: 0,
    ram: 0,
    disk: 0,
  },

  _ws: null,
  _reconnectTimer: null,

  // ── Actions ──

  /** Connect to the WebSocket telemetry endpoint */
  connect: () => {
    const { _ws } = get();
    if (_ws && _ws.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        set({ connected: true });
        console.log('[Telemetry] WebSocket connected');

        // Clear any reconnect timer
        const timer = get()._reconnectTimer;
        if (timer) {
          clearTimeout(timer);
          set({ _reconnectTimer: null });
        }
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === 'telemetry' && msg.data) {
            const d = msg.data;
            const updates = {};

            if (d.gpu) {
              updates.vramUsedMb = d.gpu.vram_used_mb || 0;
              updates.vramTotalMb = d.gpu.vram_total_mb || 0;
              updates.vramFreeMb = d.gpu.vram_free_mb || 0;
              updates.gpuTemperature = d.gpu.temperature;
            }

            if (d.ram) {
              updates.ramUsedMb = d.ram.used_mb || 0;
              updates.ramTotalMb = d.ram.total_mb || 0;
              updates.ramFreeMb = d.ram.free_mb || 0;
              updates.ramPercent = d.ram.percent || 0;
            }

            if (d.engine) {
              updates.modelLoaded = d.engine.model_loaded || false;
              updates.modelId = d.engine.model_id || null;
              updates.isGenerating = d.engine.is_generating || false;
              updates.nGpuLayers = d.engine.n_gpu_layers || 0;
              updates.contextLength = d.engine.context_length || 0;
            }

            set(updates);
          }

          if (msg.type === 'generation_progress') {
            set({
              tokensPerSec: msg.tokens_per_sec || 0,
              ttftMs: msg.ttft_ms || 0,
            });
          }

        } catch (e) {
          // Skip invalid messages
        }
      };

      ws.onclose = () => {
        set({ connected: false });
        console.log('[Telemetry] WebSocket disconnected, reconnecting in 3s...');

        // Auto-reconnect
        const timer = setTimeout(() => {
          get().connect();
        }, 3000);
        set({ _reconnectTimer: timer });
      };

      ws.onerror = (err) => {
        console.error('[Telemetry] WebSocket error:', err);
        ws.close();
      };

      set({ _ws: ws });

    } catch (err) {
      console.error('[Telemetry] Connection error:', err);
      // Retry in 3s
      const timer = setTimeout(() => get().connect(), 3000);
      set({ _reconnectTimer: timer });
    }
  },

  /** Disconnect the WebSocket */
  disconnect: () => {
    const { _ws, _reconnectTimer } = get();
    if (_reconnectTimer) clearTimeout(_reconnectTimer);
    if (_ws) _ws.close();
    set({ _ws: null, connected: false, _reconnectTimer: null });
  },

  /** Send a command through the WebSocket */
  send: (data) => {
    const { _ws } = get();
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(JSON.stringify(data));
    }
  },

  /** Update layer distribution from offload plan */
  setLayerDistribution: (gpu, ram, disk) => {
    set({
      layerDistribution: { gpu, ram, disk },
    });
  },

  /** Update generation speed metrics */
  updateGenerationMetrics: (tokensPerSec, ttftMs) => {
    set({ tokensPerSec, ttftMs });
  },
}));

export default useTelemetryStore;
