/**
 * AI Runner — SettingsModal
 * Settings form with tabs: General, Storage, Advanced, API.
 * Implements FR-601–FR-604, FR-407.
 */

import { useState, useEffect } from 'react';
import useSettingsStore from '../store/useSettingsStore';
import { useTranslation } from '../i18n/useTranslation';
import HardwareCard from './HardwareCard';
import './SettingsModal.css';

export default function SettingsModal({ isOpen, onClose }) {
  const t = useTranslation();
  const settings = useSettingsStore();
  const [activeTab, setActiveTab] = useState('general');
  const [saved, setSaved] = useState(false);
  const [localSettings, setLocalSettings] = useState({});

  useEffect(() => {
    if (isOpen) {
      setLocalSettings({
        theme: settings.theme,
        language: settings.language,
        defaultSystemPrompt: settings.defaultSystemPrompt,
        modelDir: settings.modelDir,
        cacheSizeLimitGb: settings.cacheSizeLimitGb,
        nThreads: settings.nThreads,
        useMmap: settings.useMmap,
        nBatch: settings.nBatch,
        apiHost: settings.apiHost,
        apiPort: settings.apiPort,
        apiKey: settings.apiKey,
        advancedMode: settings.advancedMode,
      });
    }
  }, [isOpen, settings]);

  const handleSave = () => {
    settings.saveSettings(localSettings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const update = (key, value) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  if (!isOpen) return null;

  const tabs = [
    { id: 'general', label: t('settings.general'), icon: '⚙️' },
    { id: 'storage', label: t('settings.storage'), icon: '💾' },
    { id: 'advanced', label: t('settings.advanced'), icon: '🔧' },
    { id: 'api', label: t('settings.api'), icon: '🔌' },
  ];

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal animate-scale-in" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="settings-header">
          <h2 className="text-heading">{t('settings.title')}</h2>
          <button className="btn btn-ghost btn-icon" onClick={onClose}>✕</button>
        </div>

        {/* Tabs */}
        <div className="settings-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`settings-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="settings-content">
          {/* General Tab */}
          {activeTab === 'general' && (
            <div className="settings-section animate-fade-in">
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.theme')}</label>
                </div>
                <div className="theme-toggle">
                  <button
                    className={`theme-btn ${localSettings.theme === 'dark' ? 'active' : ''}`}
                    onClick={() => update('theme', 'dark')}
                  >
                    🌙 {t('settings.theme_dark')}
                  </button>
                  <button
                    className={`theme-btn ${localSettings.theme === 'light' ? 'active' : ''}`}
                    onClick={() => update('theme', 'light')}
                  >
                    ☀️ {t('settings.theme_light')}
                  </button>
                </div>
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.language')}</label>
                </div>
                <div className="theme-toggle">
                  <button
                    className={`theme-btn ${localSettings.language === 'tr' ? 'active' : ''}`}
                    onClick={() => update('language', 'tr')}
                  >
                    🇹🇷 Türkçe
                  </button>
                  <button
                    className={`theme-btn ${localSettings.language === 'en' ? 'active' : ''}`}
                    onClick={() => update('language', 'en')}
                  >
                    🇬🇧 English
                  </button>
                </div>
              </div>

              <div className="setting-row setting-row-vertical">
                <label>{t('settings.system_prompt')}</label>
                <textarea
                  className="setting-textarea"
                  value={localSettings.defaultSystemPrompt || ''}
                  onChange={(e) => update('defaultSystemPrompt', e.target.value)}
                  rows={3}
                  placeholder="Varsayılan sistem promptu..."
                />
              </div>

              <div className="setting-divider" />
              <HardwareCard />
            </div>
          )}

          {/* Storage Tab */}
          {activeTab === 'storage' && (
            <div className="settings-section animate-fade-in">
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.model_dir')}</label>
                </div>
                <input
                  className="setting-input"
                  value={localSettings.modelDir || ''}
                  onChange={(e) => update('modelDir', e.target.value)}
                />
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.cache_limit')}</label>
                </div>
                <div className="setting-input-group">
                  <input
                    type="number"
                    className="setting-input-small"
                    value={localSettings.cacheSizeLimitGb || 50}
                    onChange={(e) => update('cacheSizeLimitGb', parseInt(e.target.value) || 50)}
                  />
                  <span className="text-small">GB</span>
                </div>
              </div>

              <div className="setting-divider" />

              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.export')}</label>
                </div>
                <button className="btn btn-secondary btn-sm" onClick={settings.exportSettings}>
                  📤 {t('settings.export')}
                </button>
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.import')}</label>
                </div>
                <label className="btn btn-secondary btn-sm import-btn">
                  📥 {t('settings.import')}
                  <input
                    type="file"
                    accept=".json"
                    style={{ display: 'none' }}
                    onChange={(e) => {
                      if (e.target.files?.[0]) settings.importSettings(e.target.files[0]);
                    }}
                  />
                </label>
              </div>
            </div>
          )}

          {/* Advanced Tab */}
          {activeTab === 'advanced' && (
            <div className="settings-section animate-fade-in">

              {/* Advanced Mode toggle */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.advanced_mode')}</label>
                  <span className="text-small">{t('settings.advanced_mode_desc')}</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.advancedMode || false}
                    onChange={(e) => update('advancedMode', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              <div className="setting-divider" />
              <p className="settings-section-title">⚡ Performans Optimizasyonları</p>

              {/* KV Cache Quantization */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>🗜️ KV Önbellek Tipi</label>
                  <span className="text-small">
                    Düşük bit = daha az VRAM kullanımı. <strong>q4_0</strong> önerilir (%50 tasarruf).
                  </span>
                </div>
                <select
                  className="setting-select"
                  value={localSettings.kvCacheType || 'q4_0'}
                  onChange={(e) => update('kvCacheType', e.target.value)}
                >
                  <option value="q4_0">4-bit / q4_0 — Minimum VRAM (%50 tasarruf) ⚡</option>
                  <option value="q5_0">5-bit / q5_0 — Dengeli</option>
                  <option value="q5_1">5-bit / q5_1 — Dengeli+</option>
                  <option value="q8_0">8-bit / q8_0 — Yüksek Kalite</option>
                  <option value="f16">16-bit / f16 — Varsayılan (max VRAM)</option>
                </select>
              </div>

              {/* Flash Attention */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>⚡ Flash Attention</label>
                  <span className="text-small">
                    Uzun bağlamlarda %20–40 hız artışı. CUDA/Metal gerektirir.
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.flashAttn ?? true}
                    onChange={(e) => update('flashAttn', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              {/* Memory Lock */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>🔒 Bellek Kilitleme (mlock)</label>
                  <span className="text-small">
                    İşletim sisteminin model ağırlıklarını diske (swap) yazmasını engeller.
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.useMlock ?? true}
                    onChange={(e) => update('useMlock', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              {/* Context Shifting */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>🔄 Akıllı Bağlam Kaydırma</label>
                  <span className="text-small">
                    Bağlam dolduğunda eski mesajları kırparak tam yeniden hesaplamayı önler.
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.cacheContextShift ?? true}
                    onChange={(e) => update('cacheContextShift', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              {/* Memory Mapped I/O */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>🗂️ Bellek Eşlemeli G/Ç (mmap)</label>
                  <span className="text-small">
                    Modeli diske eşleyerek RAM kullanımını azaltır (SSD önerilir).
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.useMmap ?? true}
                    onChange={(e) => update('useMmap', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              <div className="setting-divider" />
              <p className="settings-section-title">🔧 Düşük Seviye Ayarlar</p>

              {/* CPU Threads */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.threads')}</label>
                  <span className="text-small">
                    Boş bırakın = sadece fiziksel çekirdekler kullanılır (önerilir).
                  </span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.nThreads || ''}
                  onChange={(e) => update('nThreads', e.target.value ? parseInt(e.target.value) : null)}
                  placeholder="Otomatik"
                  min={1}
                  max={64}
                />
              </div>

              {/* Batch Size */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.batch_size')}</label>
                  <span className="text-small">
                    Prompt işleme batch boyutu. Büyük değer = hızlı ön işleme, daha fazla VRAM.
                  </span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.nBatch || 512}
                  onChange={(e) => update('nBatch', parseInt(e.target.value) || 512)}
                  step={64}
                  min={64}
                  max={4096}
                />
              </div>

              <div className="setting-divider" />
              <p className="settings-section-title">🚀 Spekülatif Çözme (Speculative Decoding)</p>
              <p className="text-small" style={{ marginBottom: '0.75rem', opacity: 0.7 }}>
                Büyük modelin yanına küçük bir taslak model ekleyerek 2–3x hız artışı sağlar.
                Taslak model (örn. 1B–3B) hızlıca token tahmin eder, büyük model onaylar.
              </p>

              {/* Draft Model Path */}
              <div className="setting-row setting-row-vertical">
                <label>🤏 Taslak Model Yolu (.gguf)</label>
                <input
                  className="setting-input"
                  value={localSettings.draftModelPath || ''}
                  onChange={(e) => update('draftModelPath', e.target.value)}
                  placeholder="C:\models\llama-3-1b.Q4_K_M.gguf (opsiyonel)"
                />
              </div>

              {/* Draft GPU Layers */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>GPU Katmanları (Taslak)</label>
                  <span className="text-small">-1 = tümü GPU'ya</span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.draftNGpuLayers ?? -1}
                  onChange={(e) => update('draftNGpuLayers', parseInt(e.target.value))}
                  min={-1}
                />
              </div>

            </div>
          )}


          {/* API Tab */}
          {activeTab === 'api' && (
            <div className="settings-section animate-fade-in">
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.api_host')}</label>
                </div>
                <input
                  className="setting-input"
                  value={localSettings.apiHost || '127.0.0.1'}
                  onChange={(e) => update('apiHost', e.target.value)}
                />
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.api_port')}</label>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.apiPort || 8420}
                  onChange={(e) => update('apiPort', parseInt(e.target.value) || 8420)}
                />
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <label>{t('settings.api_key')}</label>
                </div>
                <input
                  className="setting-input"
                  type="password"
                  value={localSettings.apiKey || ''}
                  onChange={(e) => update('apiKey', e.target.value || null)}
                  placeholder="Opsiyonel"
                />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="settings-footer">
          <button className="btn btn-ghost" onClick={onClose}>
            {t('settings.cancel')}
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            {saved ? '✓ ' + t('settings.saved') : t('settings.save')}
          </button>
        </div>
      </div>
    </div>
  );
}
