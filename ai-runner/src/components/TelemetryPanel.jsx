/**
 * AI Runner — TelemetryPanel
 * Right panel: live system metrics (VRAM, RAM, tokens/sec, GPU temp, layer distribution).
 * Implements FR-402, FR-403.
 */

import { useEffect } from 'react';
import useTelemetryStore from '../store/useTelemetryStore';
import useModelStore from '../store/useModelStore';
import { useTranslation } from '../i18n/useTranslation';
import LayerDistributionBar from './LayerDistributionBar';
import './TelemetryPanel.css';

function MetricBar({ label, value, max, unit, color = 'var(--accent-active)' }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const displayValue = value >= 1024
    ? `${(value / 1024).toFixed(1)} GB`
    : `${Math.round(value)} ${unit}`;
  const displayMax = max >= 1024
    ? `${(max / 1024).toFixed(1)} GB`
    : `${Math.round(max)} ${unit}`;

  return (
    <div className="metric-item">
      <div className="metric-header">
        <span className="text-small">{label}</span>
        <span className="text-mono metric-value">{displayValue} / {displayMax}</span>
      </div>
      <div className="metric-bar">
        <div
          className="metric-bar-fill"
          style={{ width: `${pct}%`, background: pct > 90 ? 'var(--accent-warning)' : color }}
        />
      </div>
    </div>
  );
}

function MetricValue({ label, value, unit, icon }) {
  return (
    <div className="metric-value-item">
      <span className="metric-icon">{icon}</span>
      <div className="metric-value-content">
        <span className="text-mono metric-number">{value}</span>
        <span className="text-small">{unit} · {label}</span>
      </div>
    </div>
  );
}

export default function TelemetryPanel({ collapsed = false }) {
  const t = useTranslation();
  const {
    connected, connect,
    vramUsedMb, vramTotalMb,
    ramUsedMb, ramTotalMb, ramPercent,
    gpuTemperature,
    tokensPerSec, ttftMs,
    modelLoaded, modelId,
    nGpuLayers, contextLength,
    layerDistribution,
  } = useTelemetryStore();

  const activeModel = useModelStore((s) => s.activeModel);

  // Connect WebSocket on mount
  useEffect(() => {
    connect();
    return () => {}; // Keep connection alive across re-renders
  }, [connect]);

  if (collapsed) return null;

  return (
    <aside className="telemetry-panel" id="telemetry-panel">
      {/* Header */}
      <div className="telemetry-header">
        <h3 className="text-subheading">{t('telemetry.title')}</h3>
        <div className={`connection-dot ${connected ? 'connected' : 'disconnected'}`}>
          <span className="text-small">
            {connected ? t('telemetry.connected') : t('telemetry.disconnected')}
          </span>
        </div>
      </div>

      <div className="telemetry-content">
        {/* VRAM Usage */}
        {vramTotalMb > 0 && (
          <MetricBar
            label={t('telemetry.vram')}
            value={vramUsedMb}
            max={vramTotalMb}
            unit="MB"
            color="var(--accent-active)"
          />
        )}

        {/* RAM Usage */}
        <MetricBar
          label={t('telemetry.ram')}
          value={ramUsedMb}
          max={ramTotalMb}
          unit="MB"
          color="var(--accent-ready)"
        />

        <div className="telemetry-divider" />

        {/* Speed Metrics */}
        <div className="metric-values-grid">
          <MetricValue
            label={t('telemetry.tokens_sec')}
            value={tokensPerSec > 0 ? tokensPerSec.toFixed(1) : '—'}
            unit={t('chat.tokens_per_sec')}
            icon="⚡"
          />
          <MetricValue
            label={t('telemetry.ttft')}
            value={ttftMs > 0 ? `${Math.round(ttftMs)}` : '—'}
            unit="ms"
            icon="🎯"
          />
          {gpuTemperature !== null && gpuTemperature !== undefined && (
            <MetricValue
              label={t('telemetry.gpu_temp')}
              value={`${gpuTemperature}`}
              unit="°C"
              icon="🌡️"
            />
          )}
        </div>

        <div className="telemetry-divider" />

        {/* Layer Distribution */}
        {modelLoaded ? (
          <LayerDistributionBar
            gpuLayers={layerDistribution.gpu || nGpuLayers || 0}
            ramLayers={layerDistribution.ram || 0}
            diskLayers={layerDistribution.disk || 0}
          />
        ) : (
          <div className="telemetry-empty">
            <span className="telemetry-empty-icon">📊</span>
            <span className="text-small">{t('telemetry.no_model')}</span>
          </div>
        )}

        {/* Active Model Info */}
        {modelLoaded && modelId && (
          <>
            <div className="telemetry-divider" />
            <div className="active-model-info">
              <span className="text-small" style={{ color: 'var(--text-tertiary)' }}>Aktif Model</span>
              <span className="text-mono" style={{ fontSize: '12px' }}>{modelId.split('/').pop()}</span>
              {contextLength > 0 && (
                <span className="text-small" style={{ color: 'var(--text-tertiary)' }}>
                  Context: {contextLength.toLocaleString()}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
