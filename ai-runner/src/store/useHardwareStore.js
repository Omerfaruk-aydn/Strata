/**
 * AI Runner — Hardware Store
 * Hardware profile data from backend.
 * Implements FR-201–FR-204.
 */

import { create } from 'zustand';

const API_BASE = 'http://127.0.0.1:8420';

const useHardwareStore = create((set, get) => ({
  // ── State ──
  profile: null,
  isLoading: false,
  vramWarning: null,
  error: null,

  // ── Actions ──

  /** FR-201: Fetch hardware profile */
  fetchProfile: async () => {
    set({ isLoading: true, error: null });
    try {
      const res = await fetch(`${API_BASE}/api/hardware/profile`);
      if (!res.ok) throw new Error('Donanım profili alınamadı');
      const data = await res.json();

      set({
        profile: data,
        vramWarning: data.vram_warning || null,
        isLoading: false,
      });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** FR-203: Manual refresh */
  refreshProfile: async () => {
    set({ isLoading: true });
    try {
      const res = await fetch(`${API_BASE}/api/hardware/refresh`, { method: 'POST' });
      if (!res.ok) throw new Error('Profil yenilenemedi');
      const data = await res.json();
      set({
        profile: data,
        vramWarning: data.vram_warning || null,
        isLoading: false,
      });
    } catch (err) {
      set({ error: err.message, isLoading: false });
    }
  },

  /** FR-202: Select GPU (multi-GPU) */
  selectGpu: (index) => {
    set((state) => {
      if (!state.profile?.gpus?.[index]) return {};
      return {
        profile: {
          ...state.profile,
          gpu: state.profile.gpus[index],
          selected_gpu_index: index,
        },
      };
    });
  },

  clearWarning: () => set({ vramWarning: null }),
  clearError: () => set({ error: null }),
}));

export default useHardwareStore;
