/**
 * AI Runner — LayerDistributionBar
 * Signature UI element: segmented horizontal bar showing GPU/RAM/disk layer distribution.
 * Section 13: amber=GPU, teal=RAM, grey-blue=disk. 300ms animated transitions.
 */

import { useState } from 'react';
import { useTranslation } from '../i18n/useTranslation';
import './LayerDistributionBar.css';

export default function LayerDistributionBar({ gpuLayers = 0, ramLayers = 0, diskLayers = 0 }) {
  const t = useTranslation();
  const [hoveredSegment, setHoveredSegment] = useState(null);

  const total = gpuLayers + ramLayers + diskLayers;
  if (total === 0) return null;

  const gpuPct = (gpuLayers / total) * 100;
  const ramPct = (ramLayers / total) * 100;
  const diskPct = (diskLayers / total) * 100;

  const segments = [
    { key: 'gpu', layers: gpuLayers, pct: gpuPct, label: t('telemetry.gpu_label'), className: 'segment-gpu' },
    { key: 'ram', layers: ramLayers, pct: ramPct, label: t('telemetry.ram_label'), className: 'segment-ram' },
    { key: 'disk', layers: diskLayers, pct: diskPct, label: t('telemetry.disk_label'), className: 'segment-disk' },
  ].filter(s => s.layers > 0);

  return (
    <div className="layer-bar-container">
      <div className="layer-bar-header">
        <span className="text-small">{t('telemetry.layer_dist')}</span>
        <span className="text-mono layer-bar-total">{total} {t('telemetry.layers')}</span>
      </div>

      <div className="layer-bar" role="meter" aria-label={t('telemetry.layer_dist')}>
        {segments.map((seg) => (
          <div
            key={seg.key}
            className={`layer-bar-segment ${seg.className}`}
            style={{ width: `${seg.pct}%` }}
            onMouseEnter={() => setHoveredSegment(seg.key)}
            onMouseLeave={() => setHoveredSegment(null)}
          >
            {seg.pct > 12 && (
              <span className="segment-label">{seg.layers}</span>
            )}

            {hoveredSegment === seg.key && (
              <div className="segment-tooltip">
                <strong>{seg.label}</strong>
                <span>{seg.layers} {t('telemetry.layers')}</span>
                <span>{seg.pct.toFixed(1)}%</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="layer-bar-legend">
        {segments.map((seg) => (
          <div key={seg.key} className="legend-item">
            <div className={`legend-dot ${seg.className}`} />
            <span className="text-small">{seg.label}</span>
            <span className="text-mono legend-count">{seg.layers}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
