/**
 * AI Runner — System Optimizer Store
 * Manages state for pagefile analysis, service audit, RAM disk, and process list.
 */

import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8420';

const useOptimizerStore = create((set, get) => ({
  // ── State ──
  status: null,         // SystemOptimizerStatus
  pagefile: null,       // PagefileInfo
  services: [],         // ServiceInfo[]
  processes: [],        // ProcessInfo[]
  ramdisk: null,        // RamDiskInfo
  isLoading: false,
  error: null,
  lastFetched: null,
  copiedCommand: null,  // Which command was just copied (for UI feedback)

  // ── Actions ──

  /** Load overall system optimization status */
  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      const res = await fetch(`${API_BASE}/api/optimizer/status`);
      if (!res.ok) throw new Error('Sistem durumu alınamadı');
      const data = await res.json();
      set({ status: data, isLoading: false, lastFetched: new Date() });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** ① Fetch pagefile analysis */
  fetchPagefile: async (modelSizeMb = 0) => {
    set({ isLoading: true });
    try {
      const res = await fetch(`${API_BASE}/api/optimizer/pagefile?model_size_mb=${modelSizeMb}`);
      if (!res.ok) throw new Error('Pagefile bilgisi alınamadı');
      const data = await res.json();
      set({ pagefile: data, isLoading: false });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** ② Fetch service audit + top processes */
  fetchServices: async () => {
    set({ isLoading: true });
    try {
      const [svcRes, procRes] = await Promise.all([
        fetch(`${API_BASE}/api/optimizer/services`),
        fetch(`${API_BASE}/api/optimizer/processes?limit=10`),
      ]);
      const svcData = svcRes.ok ? await svcRes.json() : { services: [] };
      const procData = procRes.ok ? await procRes.json() : { processes: [] };
      set({
        services: svcData.services || [],
        processes: procData.processes || [],
        isLoading: false,
      });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** ③ Fetch RAM disk feasibility */
  fetchRamdisk: async (modelSizeMb = 0) => {
    set({ isLoading: true });
    try {
      const res = await fetch(`${API_BASE}/api/optimizer/ramdisk?model_size_mb=${modelSizeMb}`);
      if (!res.ok) throw new Error('RAM Disk bilgisi alınamadı');
      const data = await res.json();
      set({ ramdisk: data, isLoading: false });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** Fetch all data at once */
  fetchAll: async (modelSizeMb = 0) => {
    await Promise.all([
      get().fetchStatus(),
      get().fetchPagefile(modelSizeMb),
      get().fetchServices(),
      get().fetchRamdisk(modelSizeMb),
    ]);
  },

  /** Copy a command to clipboard and show feedback */
  copyCommand: async (command, key) => {
    try {
      await navigator.clipboard.writeText(command);
      set({ copiedCommand: key });
      setTimeout(() => set({ copiedCommand: null }), 2500);
    } catch {
      // Fallback: create temporary textarea
      const ta = document.createElement('textarea');
      ta.value = command;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      set({ copiedCommand: key });
      setTimeout(() => set({ copiedCommand: null }), 2500);
    }
  },

  clearError: () => set({ error: null }),
}));

export default useOptimizerStore;
