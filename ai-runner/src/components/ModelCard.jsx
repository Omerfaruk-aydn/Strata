/**
 * AI Runner — ModelCard
 * Individual model summary with compatibility badge, size info, and actions.
 * Implements FR-103, FR-104.
 */

import { useState } from 'react';
import { useTranslation } from '../i18n/useTranslation';
import useModelStore from '../store/useModelStore';
import useSettingsStore from '../store/useSettingsStore';
import './ModelCard.css';

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '—';
  const gb = bytes / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / (1024 ** 2);
  return `${mb.toFixed(0)} MB`;
}

function formatNumber(num) {
  if (!num) return '—';
  if (num >= 1e9) return `${(num / 1e9).toFixed(0)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(0)}M`;
  return num.toLocaleString();
}

const BADGE_MAP = {
  compatible: { emoji: '🟢', className: 'badge-compatible' },
  limited: { emoji: '🟡', className: 'badge-limited' },
  incompatible: { emoji: '🔴', className: 'badge-incompatible' },
};

export default function ModelCard({ model, isLocal = false, onSelect }) {
  const t = useTranslation();
  const { downloadModel, loadModel, unloadModel, deleteModel, activeModel, downloadProgress, loadingModelId } = useModelStore();
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

  const badge = BADGE_MAP[model.compatibility] || BADGE_MAP.compatible;
  const isActive = activeModel?.model_id === model.id;
  const isLoading = loadingModelId === model.id;
  const progress = downloadProgress[model.id];
  const isDownloading = progress?.status === 'downloading';

  const handleLoad = async () => {
    try {
      const settings = useSettingsStore.getState();
      await loadModel(model.id, {
        quant:              model.downloaded_quant || 'Q4_K_M',
        contextLength:      settings.maxContextLength,
        nThreads:           settings.nThreads,
        nBatch:             settings.nBatch,
        useMmap:            settings.useMmap,
        useMlock:           settings.useMlock,
        kvCacheType:        settings.kvCacheType,
        flashAttn:          settings.flashAttn,
        cacheContextShift:  settings.cacheContextShift,
        draftModelPath:     settings.draftModelPath,
        draftNGpuLayers:    settings.draftNGpuLayers,
      });
    } catch (e) { /* handled in store */ }
  };

  const handleDelete = () => {
    if (showConfirmDelete) {
      deleteModel(model.id);
      setShowConfirmDelete(false);
    } else {
      setShowConfirmDelete(true);
      setTimeout(() => setShowConfirmDelete(false), 3000);
    }
  };

  return (
    <div
      className={`model-card ${isActive ? 'model-card-active' : ''}`}
      onClick={() => onSelect?.(model)}
      role="button"
      tabIndex={0}
      id={`model-card-${model.id.replace(/[^a-zA-Z0-9]/g, '-')}`}
    >
      {/* Header */}
      <div className="model-card-header">
        <div className="model-card-title-row">
          <h4 className="model-card-name truncate">{model.display_name}</h4>
          <span className={`badge ${badge.className}`}>
            {badge.emoji} {t(`models.${model.compatibility || 'compatible'}`)}
          </span>
        </div>
        {model.author && (
          <span className="model-card-author text-small">{model.author}</span>
        )}
      </div>

      {/* Stats */}
      <div className="model-card-stats">
        {model.parameter_count > 0 && (
          <div className="stat-chip">
            <span className="stat-label">{t('models.params')}</span>
            <span className="stat-value">{formatNumber(model.parameter_count)}</span>
          </div>
        )}
        {model.file_size_bytes > 0 && (
          <div className="stat-chip">
            <span className="stat-label">{t('models.size')}</span>
            <span className="stat-value">{formatBytes(model.file_size_bytes)}</span>
          </div>
        )}
        {model.downloaded_quant && (
          <div className="stat-chip">
            <span className="stat-label">{t('models.quant')}</span>
            <span className="stat-value">{model.downloaded_quant}</span>
          </div>
        )}
        {model.context_length > 0 && (
          <div className="stat-chip">
            <span className="stat-label">{t('models.context')}</span>
            <span className="stat-value">{(model.context_length / 1024).toFixed(0)}K</span>
          </div>
        )}
      </div>

      {/* Download Progress */}
      {isDownloading && (
        <div className="model-card-progress">
          <div className="progress-bar">
            <div
              className="progress-bar-fill"
              style={{ width: `${(progress.progress || 0) * 100}%` }}
            />
          </div>
          <div className="progress-info">
            <span className="text-small">{Math.round((progress.progress || 0) * 100)}%</span>
            {progress.speed > 0 && (
              <span className="text-small">{progress.speed.toFixed(1)} MB/s</span>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="model-card-actions">
        {isLocal ? (
          <>
            {isActive ? (
              <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); unloadModel(); }}>
                {t('models.unload')}
              </button>
            ) : (
              <button
                className="btn btn-primary btn-sm"
                onClick={(e) => { e.stopPropagation(); handleLoad(); }}
                disabled={isLoading}
              >
                {isLoading ? t('models.loading') : t('models.load')}
              </button>
            )}
            <button
              className={`btn btn-sm ${showConfirmDelete ? 'btn-danger' : 'btn-ghost'}`}
              onClick={(e) => { e.stopPropagation(); handleDelete(); }}
            >
              {showConfirmDelete ? t('models.delete_confirm').split('?')[0] + '?' : t('models.delete')}
            </button>
          </>
        ) : (
          <button
            className="btn btn-primary btn-sm"
            onClick={(e) => {
              e.stopPropagation();
              downloadModel(model.id, model.available_quants?.[1] || 'Q4_K_M');
            }}
            disabled={isDownloading}
          >
            {isDownloading ? t('models.downloading') : t('models.download')}
          </button>
        )}
      </div>

      {/* Active indicator */}
      {isActive && <div className="model-card-active-indicator" />}
    </div>
  );
}
