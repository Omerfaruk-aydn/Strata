/**
 * AI Runner — Main Application
 * Three-panel layout: ModelShelf (left) | ChatConsole (center) | TelemetryPanel (right)
 * Implements FR-401, FR-406 (keyboard shortcuts), FR-407 (theme).
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import useSettingsStore from './store/useSettingsStore';
import useHardwareStore from './store/useHardwareStore';
import useSessionStore from './store/useSessionStore';
import useModelStore from './store/useModelStore';
import { useTranslation } from './i18n/useTranslation';
import ModelShelf from './components/ModelShelf';
import ChatConsole from './components/ChatConsole';
import TelemetryPanel from './components/TelemetryPanel';
import SessionList from './components/SessionList';
import SettingsModal from './components/SettingsModal';
import SystemOptimizer from './components/SystemOptimizer';
import ExtremeModelCenter from './components/ExtremeModelCenter';
import { configureApi, waitForApi } from './api/client';
import './App.css';

export default function App() {
  const t = useTranslation();
  const [leftCollapsed, setLeftCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [optimizerOpen, setOptimizerOpen] = useState(false);
  const [extremeOpen, setExtremeOpen] = useState(false);
  const [showSessions, setShowSessions] = useState(true);
  const [startupState, setStartupState] = useState('starting');
  const [startupError, setStartupError] = useState('');
  const [startupApiKey, setStartupApiKey] = useState('');
  const initializationStarted = useRef(false);

  const { fetchSettings, theme } = useSettingsStore();
  const { fetchProfile } = useHardwareStore();
  const { fetchSessions, createSession } = useSessionStore();
  const { fetchLocalModels } = useModelStore();
  const error = useSessionStore((s) => s.error);
  const modelError = useModelStore((s) => s.error);

  const initialize = useCallback(async () => {
    setStartupState('starting');
    setStartupError('');
    try {
      await waitForApi();
      const settingsLoaded = await fetchSettings();
      if (!settingsLoaded) {
        throw new Error(useSettingsStore.getState().error || 'Ayarlar okunamadı.');
      }
      await Promise.all([fetchProfile(), fetchSessions(), fetchLocalModels()]);
      setStartupState('ready');
    } catch (startupFailure) {
      setStartupError(startupFailure.message || 'Backend bağlantısı kurulamadı.');
      setStartupState('error');
    }
  }, [fetchLocalModels, fetchProfile, fetchSessions, fetchSettings]);

  // ── Initialize on mount ──
  useEffect(() => {
    if (initializationStarted.current) return;
    initializationStarted.current = true;
    initialize();
  }, [initialize]);

  // ── Keyboard Shortcuts (FR-406) ──
  const handleKeyDown = useCallback((e) => {
    const isMod = e.ctrlKey || e.metaKey;

    // Ctrl/Cmd + N: New chat
    if (isMod && e.key === 'n') {
      e.preventDefault();
      createSession();
    }

    // Ctrl/Cmd + K: Focus search
    if (isMod && e.key === 'k') {
      e.preventDefault();
      document.getElementById('model-search-input')?.focus();
    }

    // Ctrl/Cmd + ,: Open settings
    if (isMod && e.key === ',') {
      e.preventDefault();
      setSettingsOpen(true);
    }

    // Ctrl/Cmd + Shift + O: Open System Optimizer
    if (isMod && e.shiftKey && e.key.toLowerCase() === 'o') {
      e.preventDefault();
      setOptimizerOpen(true);
    }

    // Ctrl/Cmd + Shift + E: Open Extreme Model Center
    if (isMod && e.shiftKey && e.key.toLowerCase() === 'e') {
      e.preventDefault();
      setExtremeOpen(true);
    }

    // Ctrl/Cmd + B: Toggle left sidebar
    if (isMod && e.key === 'b') {
      e.preventDefault();
      setLeftCollapsed((prev) => !prev);
    }

    // Ctrl/Cmd + .: Toggle right sidebar
    if (isMod && e.key === '.') {
      e.preventDefault();
      setRightCollapsed((prev) => !prev);
    }
  }, [createSession]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (startupState !== 'ready') {
    const retry = () => {
      if (startupApiKey.trim()) {
        configureApi({ apiKey: startupApiKey.trim() });
      }
      useSettingsStore.getState().clearError();
      initialize();
    };

    return (
      <div className="app app-startup" data-theme={theme}>
        <div className="startup-mark" aria-hidden="true">A</div>
        <h1>AI Runner</h1>
        {startupState === 'starting' ? (
          <>
            <div className="startup-spinner" aria-label="Yükleniyor" />
            <p>Yerel yapay zekâ motoru hazırlanıyor…</p>
            <span>İlk açılışta paket çıkarma işlemi biraz sürebilir.</span>
          </>
        ) : (
          <div className="startup-recovery">
            <p>{startupError}</p>
            <label htmlFor="startup-api-key">API anahtarı (gerekiyorsa)</label>
            <input
              id="startup-api-key"
              className="setting-input"
              type="password"
              value={startupApiKey}
              onChange={(event) => setStartupApiKey(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && retry()}
              autoComplete="current-password"
            />
            <button className="btn btn-primary" onClick={retry}>Yeniden Dene</button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="app" data-theme={theme}>
      {/* ── Header Bar ── */}
      <header className="app-header">
        <div className="header-left">
          <button
            className="btn btn-ghost btn-icon header-toggle"
            onClick={() => setLeftCollapsed(!leftCollapsed)}
            title="Model Rafı (Ctrl+B)"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M2 4h14M2 9h14M2 14h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
          <div className="header-brand">
            <h1 className="header-title">AI Runner</h1>
            <span className="header-version">v1.0</span>
          </div>
        </div>

        <div className="header-center">
          <button
            className={`btn btn-ghost btn-sm ${showSessions ? 'active' : ''}`}
            onClick={() => setShowSessions(!showSessions)}
          >
            💬 {t('chat.sessions')}
          </button>
          <button
            className="btn btn-ghost btn-sm header-extreme-button"
            onClick={() => setExtremeOpen(true)}
            title="Extreme Model Center (Ctrl+Shift+E)"
          >
            <span aria-hidden="true">◆</span> Extreme Mode
          </button>
        </div>

        <div className="header-right">
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setOptimizerOpen(true)}
            title="Sistem Optimizasyonu (Ctrl+Shift+O)"
            style={{ marginRight: '4px' }}
          >
            🔧
          </button>
          <button
            className="btn btn-ghost btn-icon"
            onClick={() => setSettingsOpen(true)}
            title={t('settings.title') + ' (Ctrl+,)'}
          >
            ⚙️
          </button>
          <button
            className="btn btn-ghost btn-icon header-toggle"
            onClick={() => setRightCollapsed(!rightCollapsed)}
            title="Telemetri (Ctrl+.)"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M3 14V8M7 14V4M11 14V10M15 14V6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      </header>

      {/* ── Main Content ── */}
      <div className="app-content">
        {/* Left: Model Shelf */}
        {!leftCollapsed && (
          <div className="left-panel">
            {showSessions && (
              <div className="sessions-section">
                <SessionList />
              </div>
            )}
            <div className="models-section">
              <ModelShelf />
            </div>
          </div>
        )}

        {/* Center: Chat Console */}
        <ChatConsole />

        {/* Right: Telemetry Panel */}
        <TelemetryPanel collapsed={rightCollapsed} />
      </div>

      {/* ── Settings Modal ── */}
      <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* ── System Optimizer Modal ── */}
      <SystemOptimizer isOpen={optimizerOpen} onClose={() => setOptimizerOpen(false)} />

      {/* ── Extreme Model Center ── */}
      <ExtremeModelCenter isOpen={extremeOpen} onClose={() => setExtremeOpen(false)} />

      {/* ── Error Toast ── */}
      {(error || modelError) && (
        <div className="toast toast-error animate-fade-in">
          <div className="toast-content">
            <span className="toast-icon">⚠️</span>
            <span className="toast-message">{error || modelError}</span>
          </div>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              useSessionStore.getState().clearError();
              useModelStore.getState().clearError();
            }}
          >
            {t('errors.dismiss')}
          </button>
        </div>
      )}
    </div>
  );
}
