/**
 * AI Runner — Hardware Store
 * Hardware profile data from backend.
 * Implements FR-201–FR-204.
 */

import { create } from 'zustand';
import { apiFetch } from '../api/client';
import useSettingsStore from './useSettingsStore';

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
      const res = await apiFetch('/api/hardware/profile');
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
      const res = await apiFetch('/api/hardware/refresh', { method: 'POST' });
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
  selectGpu: async (index) => {
    if (!get().profile?.gpus?.[index]) return;
    set((state) => {
      return {
        profile: {
          ...state.profile,
          gpu: state.profile.gpus[index],
          selected_gpu_index: index,
        },
      };
    });
    await useSettingsStore.getState().saveSettings({
      selectedGpuIndex: index,
      tensorSplit: null,
    });
  },

  clearWarning: () => set({ vramWarning: null }),
  clearError: () => set({ error: null }),
}));

export default useHardwareStore;
