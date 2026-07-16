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
        // Performance
        kvCacheType: settings.kvCacheType,
        flashAttn: settings.flashAttn,
        useMlock: settings.useMlock,
        cacheContextShift: settings.cacheContextShift,
        speculativeDecoding: settings.speculativeDecoding,
        draftNumPredTokens: settings.draftNumPredTokens,
        // Prompt Pruning
        maxContextLength: settings.maxContextLength,
        maxHistoryMessages: settings.maxHistoryMessages,
        autoContextPrune: settings.autoContextPrune,
        contextCompactionMode: settings.contextCompactionMode,
        extremeModeEnabled: settings.extremeModeEnabled,
        extremePreset: settings.extremePreset,
        adaptiveLoad: settings.adaptiveLoad,
        adaptiveMaxAttempts: settings.adaptiveMaxAttempts,
        backendPreference: settings.backendPreference,
        generationTimeoutS: settings.generationTimeoutS,
        allowNetworkAccess: settings.allowNetworkAccess,
      });
    }
  }, [isOpen, settings]);

  const handleSave = async () => {
    const success = await settings.saveSettings(localSettings);
    if (success) {
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  };

  const update = (key, value) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  if (!isOpen) return null;

  const tabs = [
    { id: 'general',  label: t('settings.general'),  icon: '⚙️' },
    { id: 'storage',  label: t('settings.storage'),  icon: '💾' },
    { id: 'advanced', label: t('settings.advanced'), icon: '🔧' },
    { id: 'extreme',  label: 'Extreme Model',         icon: '◆' },
    { id: 'pruning',  label: 'Bağlam Yönetimi',      icon: '✂️' },
    { id: 'api',      label: t('settings.api'),      icon: '🔌' },
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
                    Desteklenen CUDA/Metal arka uçlarında uzun bağlamları hızlandırabilir.
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

              {/* Context Pruning */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>🔄 Otomatik Bağlam Kırpma</label>
                  <span className="text-small">
                    Bağlam bütçesi dolduğunda eski konuşma turlarını güvenli şekilde çıkarır.
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
              <p className="settings-section-title">🚀 Spekülatif Çözme</p>
              <p className="text-small" style={{ marginBottom: '0.75rem', opacity: 0.7 }}>
                Prompt içindeki tekrarları kullanarak sonraki tokenları önceden tahmin eder.
                Ek model veya VRAM gerektirmez; özellikle kod ve tekrarlı metinlerde faydalıdır.
              </p>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Prompt Lookup Decoding</label>
                  <span className="text-small">llama-cpp-python tarafından yerel olarak desteklenir.</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.speculativeDecoding ?? false}
                    onChange={(e) => update('speculativeDecoding', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Ön Tahmin Token Sayısı</label>
                  <span className="text-small">Düşük değerler daha güvenlidir; önerilen: 10.</span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.draftNumPredTokens ?? 10}
                  onChange={(e) => update('draftNumPredTokens', parseInt(e.target.value) || 10)}
                  min={1}
                  max={64}
                />
              </div>

            </div>
          )}


          {/* Extreme Model Mode */}
          {activeTab === 'extreme' && (
            <div className="settings-section animate-fade-in">
              <p className="settings-section-title">◆ Çok Büyük Model Orkestrasyonu</p>
              <p className="text-small" style={{ marginBottom: '1rem', opacity: 0.7 }}>
                70B–200B GGUF modelleri için bellek bütçesi, GPU/CPU offload ve güvenli OOM fallback ayarları.
              </p>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Extreme Model planlayıcısı</label>
                  <span className="text-small">Manuel GPU katmanı verilmediğinde gerçek GGUF metadata ve güncel boş belleği kullanır.</span>
                </div>
                <label className="toggle-switch">
                  <input type="checkbox" checked={localSettings.extremeModeEnabled ?? true} onChange={(e) => update('extremeModeEnabled', e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>

              {/* Generation timeout */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>Generation timeout</label>
                  <span className="text-small">Model yanıtı bu süreyi aşarsa üretim kontrollü olarak durdurulur. 0 = sınırsız.</span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.generationTimeoutS ?? 300}
                  onChange={(e) => update('generationTimeoutS', Math.max(0, Math.min(86400, Number(e.target.value) || 0)))}
                  min={0}
                  max={86400}
                  step={1}
                />
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Varsayılan kapasite profili</label>
                  <span className="text-small">Maksimum Kapasite, 100B model yüklemelerinde batch ve context belleğini sınırlar.</span>
                </div>
                <select className="setting-input" value={localSettings.extremePreset || 'maximum_capacity'} onChange={(e) => update('extremePreset', e.target.value)}>
                  <option value="safe">Güvenli</option>
                  <option value="balanced">Dengeli</option>
                  <option value="performance">Performans</option>
                  <option value="maximum_capacity">Maksimum Kapasite</option>
                </select>
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Native backend tercihi</label>
                  <span className="text-small">Backend, kurulu llama.cpp derlemesiyle eşleşmelidir. Değişiklik farklı bir native runtime gerektirebilir.</span>
                </div>
                <select className="setting-input" value={localSettings.backendPreference || 'auto'} onChange={(e) => update('backendPreference', e.target.value)}>
                  <option value="auto">Otomatik algıla</option>
                  <option value="cuda">CUDA</option>
                  <option value="vulkan">Vulkan</option>
                  <option value="sycl">SYCL / oneAPI</option>
                  <option value="metal">Metal</option>
                  <option value="cpu">Yalnızca CPU</option>
                </select>
              </div>

              <div className="setting-divider" />
              <p className="settings-section-title">OOM Kurtarma</p>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Uyarlanabilir yükleme</label>
                  <span className="text-small">Bellek taşarsa mlock, GPU katmanı, batch ve context değerlerini kontrollü sırayla düşürür.</span>
                </div>
                <label className="toggle-switch">
                  <input type="checkbox" checked={localSettings.adaptiveLoad ?? true} onChange={(e) => update('adaptiveLoad', e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>

              <div className="setting-row">
                <div className="setting-info">
                  <label>Maksimum yükleme denemesi</label>
                  <span className="text-small">Her deneme önceki native model bağlamını tamamen temizler.</span>
                </div>
                <input type="number" className="setting-input-small" min="1" max="12" value={localSettings.adaptiveMaxAttempts || 6} onChange={(e) => update('adaptiveMaxAttempts', Math.max(1, Math.min(12, parseInt(e.target.value) || 6)))} />
              </div>
            </div>
          )}

          {/* Pruning Tab (FR-608 Context Management / Prompt Pruning) */}
          {activeTab === 'pruning' && (
            <div className="settings-section animate-fade-in">
              <p className="settings-section-title">✂️ Bağlam Kırpma & Yönetimi</p>
              <p className="text-small" style={{ marginBottom: '1rem', opacity: 0.7 }}>
                Uzun sohbetlerin bellek aşımı yapmasını veya performansı düşürmesini engellemek için bağlam bütçenizi kontrol edin.
              </p>

              {/* Context Length Slider */}
              <div className="setting-row setting-row-vertical">
                <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                  <label>Maksimum Bağlam Boyutu (Token)</label>
                  <span className="text-small" style={{ fontWeight: 'bold', color: 'var(--accent-primary)' }}>
                    {localSettings.maxContextLength || 4096} Token
                  </span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="16384"
                  step="512"
                  value={localSettings.maxContextLength || 4096}
                  onChange={(e) => update('maxContextLength', parseInt(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)', marginTop: '0.5rem' }}
                />
                <span className="text-small" style={{ marginTop: '0.25rem', opacity: 0.6 }}>
                  Daha düşük bağlam boyutu KV önbellek bellek kullanımını azaltır ve hızı artırır.
                </span>
              </div>

              {/* Max History Messages */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>Sohbet Geçmişi Limiti</label>
                  <span className="text-small">
                    Sadece son N mesajı model belleğinde tut (0 = limitsiz).
                  </span>
                </div>
                <input
                  type="number"
                  className="setting-input-small"
                  value={localSettings.maxHistoryMessages ?? 20}
                  onChange={(e) => update('maxHistoryMessages', parseInt(e.target.value) || 0)}
                  min="0"
                  max="100"
                />
              </div>
              {/* Auto Context Pruning Toggle */}
              <div className="setting-row">
                <div className="setting-info">
                  <label>Otomatik Kırpma (Auto-Pruning)</label>
                  <span className="text-small">
                    Geçmiş, seçilen bağlam ve yanıt bütçesine sığmadığında eski mesajları otomatik kırp.
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.autoContextPrune ?? true}
                    onChange={(e) => update('autoContextPrune', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <label>Geçmiş sıkıştırma yöntemi</label>
                  <span className="text-small">
                    Eski mesajları tamamen silmek yerine sınırlı bir özet olarak bağlama ekleyebilir.
                  </span>
                </div>
                <select
                  className="setting-input"
                  value={localSettings.contextCompactionMode || 'extractive_summary'}
                  onChange={(e) => update('contextCompactionMode', e.target.value)}
                >
                  <option value="extractive_summary">Sıkıştırılmış özet</option>
                  <option value="drop_oldest">En eski mesajları kaldır</option>
                </select>
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
                  {settings.apiKeySource === 'environment' && (
                    <span className="text-small">AI_RUNNER_API_KEY ortam değişkeni tarafından yönetiliyor.</span>
                  )}
                </div>
                <input
                  className="setting-input"
                  type="password"
                  value={localSettings.apiKey || ''}
                  onChange={(e) => update('apiKey', e.target.value || null)}
                  placeholder={settings.apiKeySource === 'environment' ? 'Ortam değişkeninden alındı' : 'Opsiyonel'}
                  disabled={settings.apiKeySource === 'environment'}
                />
              </div>
              <div className="setting-row">
                <div className="setting-info">
                  <label>Yerel Ağ Erişimine İzin Ver</label>
                  <span className="text-small">
                    Yalnızca API'yi başka cihazlara açmanız gerekiyorsa etkinleştirin. API anahtarı zorunludur;
                    değişiklik yeniden başlatmada uygulanır.
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={localSettings.allowNetworkAccess ?? false}
                    onChange={(e) => update('allowNetworkAccess', e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="settings-footer">
          {settings.error ? (
            <span className="text-small" style={{ color: 'var(--color-red)', marginRight: 'auto' }}>
              {settings.error}
            </span>
          ) : settings.restartRequired ? (
            <span className="text-small" style={{ color: 'var(--accent-primary)', marginRight: 'auto' }}>
              API adresi bir sonraki uygulama açılışında etkinleşecek.
            </span>
          ) : null}
          <button className="btn btn-ghost" onClick={onClose}>
            {t('settings.cancel')}
          </button>
          <button className="btn btn-primary" onClick={handleSave} disabled={settings.isSaving}>
            {settings.isSaving ? 'Kaydediliyor…' : (saved ? '✓ ' + t('settings.saved') : t('settings.save'))}
          </button>
        </div>
      </div>
    </div>
  );
}
