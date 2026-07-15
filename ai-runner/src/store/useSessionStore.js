/**
 * AI Runner — Session Store
 * Manages chat sessions, messages, and active session state.
 * Implements FR-404, FR-701–FR-703.
 */

import { create } from 'zustand';
import useTelemetryStore from './useTelemetryStore';
import { apiFetch, readSse } from '../api/client';

const useSessionStore = create((set, get) => ({
  // ── State ──
  sessions: [],
  activeSessionId: null,
  messages: [],
  isGenerating: false,
  streamingContent: '',
  streamingTokens: 0,
  streamingSpeed: 0,
  stopRequested: false,
  error: null,

  // ── Computed ──
  get activeSession() {
    const state = get();
    return state.sessions.find(s => s.id === state.activeSessionId) || null;
  },

  // ── Session Actions (FR-701–FR-703) ──

  /** Fetch all sessions */
  fetchSessions: async () => {
    try {
      const res = await apiFetch('/api/sessions');
      const data = await res.json();
      set({ sessions: data.sessions || [] });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** Create a new session */
  createSession: async (title = 'Yeni Sohbet', modelId = null) => {
    try {
      const res = await apiFetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, model_id: modelId }),
      });
      const session = await res.json();

      set((state) => ({
        sessions: [session, ...state.sessions],
        activeSessionId: session.id,
        messages: [],
      }));

      return session;
    } catch (err) {
      set({ error: err.message });
      return null;
    }
  },

  /** Select and load a session */
  selectSession: async (sessionId) => {
    set({ activeSessionId: sessionId, messages: [], error: null });
    try {
      const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
      const data = await res.json();
      set({ messages: data.messages || [] });
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-703: Rename a session */
  renameSession: async (sessionId, title) => {
    try {
      await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      });
      set((state) => ({
        sessions: state.sessions.map(s =>
          s.id === sessionId ? { ...s, title } : s
        ),
      }));
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-703: Pin/unpin a session */
  togglePin: async (sessionId) => {
    const session = get().sessions.find(s => s.id === sessionId);
    if (!session) return;

    const pinned = !session.pinned;
    try {
      await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned }),
      });
      set((state) => ({
        sessions: state.sessions.map(s =>
          s.id === sessionId ? { ...s, pinned } : s
        ),
      }));
    } catch (err) {
      set({ error: err.message });
    }
  },

  /** FR-703: Delete a session */
  deleteSession: async (sessionId) => {
    try {
      await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
      set((state) => {
        const sessions = state.sessions.filter(s => s.id !== sessionId);
        const activeSessionId = state.activeSessionId === sessionId
          ? (sessions[0]?.id || null)
          : state.activeSessionId;
        return { sessions, activeSessionId, messages: activeSessionId === state.activeSessionId ? state.messages : [] };
      });

      // Load new active session's messages
      const newActive = get().activeSessionId;
      if (newActive) get().selectSession(newActive);
    } catch (err) {
      set({ error: err.message });
    }
  },

  // ── Message Actions ──

  /** Send a message and receive streaming response */
  sendMessage: async (content, params = {}) => {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    // Add user message optimistically
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    set((state) => ({
      messages: [...state.messages, userMsg],
      isGenerating: true,
      streamingContent: '',
      streamingTokens: 0,
      streamingSpeed: 0,
      stopRequested: false,
      error: null,
    }));

    try {
      const res = await apiFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          session_id: activeSessionId,
          system_prompt: params.systemPrompt || '',
          temperature: params.temperature ?? 0.7,
          top_p: params.topP ?? 0.9,
          top_k: params.topK ?? 40,
          repeat_penalty: params.repeatPenalty ?? 1.1,
          max_tokens: params.maxTokens ?? 2048,
        }),
      });

      let fullContent = '';
      let receivedTerminalEvent = false;

      for await (const data of readSse(res)) {
        if (data.type === 'token') {
          fullContent += data.content;
          set({
            streamingContent: fullContent,
            streamingTokens: data.tokens_generated || 0,
            streamingSpeed: data.tokens_per_sec || 0,
          });
          useTelemetryStore.getState().updateGenerationMetrics(data.tokens_per_sec || 0, 0);
        } else if (data.type === 'done') {
          receivedTerminalEvent = true;
          const result = data.result || {};
          if (fullContent) {
            const assistantMsg = {
              id: globalThis.crypto?.randomUUID?.() || Date.now() + 1,
              role: 'assistant',
              content: fullContent,
              timestamp: new Date().toISOString(),
              tokens_generated: result.tokens_generated || 0,
            };
            set((state) => ({ messages: [...state.messages, assistantMsg] }));
          }
          set({
            isGenerating: false,
            stopRequested: false,
            streamingContent: '',
          });
          useTelemetryStore.getState().updateGenerationMetrics(
            result.tokens_per_sec || 0,
            result.ttft_ms || 0
          );
        } else if (data.type === 'error') {
          receivedTerminalEvent = true;
          throw new Error(data.error || 'Üretim sırasında bilinmeyen bir hata oluştu.');
        }
      }

      if (!receivedTerminalEvent) {
        throw new Error('Yanıt akışı tamamlanmadan bağlantı kapandı.');
      }
    } catch (err) {
      set({
        error: err.message,
        isGenerating: false,
        stopRequested: false,
        streamingContent: '',
      });
      try {
        const syncResponse = await apiFetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`);
        const session = await syncResponse.json();
        set({ messages: session.messages || [], error: err.message });
      } catch {
        // Preserve the original stream error if re-sync is unavailable.
      }
    }
  },

  /** FR-303: Stop current generation */
  stopGeneration: async () => {
    try {
      set({ stopRequested: true });
      await apiFetch('/api/chat/stop', { method: 'POST' });
    } catch (err) {
      set({ stopRequested: false, error: err.message });
    }
  },

  /** FR-405: Export session */
  exportSession: async (sessionId, format = 'markdown') => {
    try {
      const res = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/export/${format}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat_${sessionId}.${format === 'markdown' ? 'md' : 'json'}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      set({ error: err.message });
    }
  },

  clearError: () => set({ error: null }),
}));

export default useSessionStore;
