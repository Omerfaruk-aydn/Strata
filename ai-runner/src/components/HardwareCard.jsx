/**
 * AI Runner — HardwareCard
 * Hardware profile summary display.
 * Implements FR-203.
 */

import useHardwareStore from '../store/useHardwareStore';
import { useTranslation } from '../i18n/useTranslation';
import './HardwareCard.css';

export default function HardwareCard() {
  const t = useTranslation();
  const { profile, isLoading, refreshProfile, selectGpu, vramWarning, clearWarning } = useHardwareStore();

  if (!profile) {
    return (
      <div className="hardware-card skeleton" style={{ height: 120 }} />
    );
  }

  const { gpu, cpu, ram, disk } = profile;

  return (
    <div className="hardware-card">
      <div className="hw-card-header">
        <h4 className="text-subheading">{t('hardware.title')}</h4>
        <button
          className="btn btn-ghost btn-sm"
          onClick={refreshProfile}
          disabled={isLoading}
        >
          {isLoading ? '⏳' : '🔄'} {t('hardware.refresh')}
        </button>
      </div>

      {/* VRAM Warning (FR-204) */}
      {vramWarning && (
        <div className="hw-warning">
          <span>⚠️ {vramWarning}</span>
          <button className="btn btn-ghost btn-sm" onClick={clearWarning}>✕</button>
        </div>
      )}

      {profile.gpus?.length > 1 && (
        <div className="setting-row" style={{ marginBottom: 'var(--space-3)' }}>
          <div className="setting-info">
            <label>Aktif GPU</label>
            <span className="text-small">Tek GPU modunda kullanılacak ana aygıt.</span>
          </div>
          <select
            className="setting-input"
            value={profile.selected_gpu_index ?? 0}
            onChange={(event) => selectGpu(Number(event.target.value))}
          >
            {profile.gpus.map((item, index) => (
              <option key={`${item.name}-${index}`} value={index}>
                GPU {index}: {item.name}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="hw-grid">
        {/* GPU */}
        <div className="hw-section">
          <div className="hw-section-icon">🎮</div>
          <div className="hw-section-content">
            <span className="hw-label">{t('hardware.gpu')}</span>
            <span className="hw-value">{gpu.name || t('hardware.no_gpu')}</span>
            {gpu.vram_total_mb > 0 && (
              <span className="hw-detail text-mono">
                {(gpu.vram_free_mb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(1)} GB {t('hardware.free').toLowerCase()}
              </span>
            )}
          </div>
        </div>

        {/* CPU */}
        <div className="hw-section">
          <div className="hw-section-icon">⚡</div>
          <div className="hw-section-content">
            <span className="hw-label">{t('hardware.cpu')}</span>
            <span className="hw-value truncate">{cpu.name}</span>
            <span className="hw-detail text-mono">
              {cpu.cores}C / {cpu.threads}T
            </span>
          </div>
        </div>

        {/* RAM */}
        <div className="hw-section">
          <div className="hw-section-icon">🧠</div>
          <div className="hw-section-content">
            <span className="hw-label">{t('hardware.ram')}</span>
            <span className="hw-value">
              {(ram.total_mb / 1024).toFixed(0)} GB
            </span>
            <span className="hw-detail text-mono">
              {(ram.free_mb / 1024).toFixed(1)} GB {t('hardware.free').toLowerCase()}
            </span>
          </div>
        </div>

        {/* Disk */}
        <div className="hw-section">
          <div className="hw-section-icon">💾</div>
          <div className="hw-section-content">
            <span className="hw-label">{t('hardware.disk')}</span>
            <span className="hw-value">{disk.type}</span>
            <span className="hw-detail text-mono">
              {disk.free_gb} / {disk.total_gb} GB {t('hardware.free').toLowerCase()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
