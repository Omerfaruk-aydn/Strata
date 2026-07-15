/**
 * AI Runner — System Optimizer Panel
 * Three-tab panel: Pagefile | Services | RAM Disk
 * Provides actionable recommendations without requiring elevated privileges.
 */

import { useState, useEffect } from 'react';
import useOptimizerStore from '../store/useOptimizerStore';
import './SystemOptimizer.css';

const RISK_META = {
  safe:     { label: 'Güvenli',   color: 'var(--color-green)',  icon: '✓' },
  moderate: { label: 'Orta',      color: 'var(--color-amber)',  icon: '⚠' },
  caution:  { label: 'Dikkatli',  color: 'var(--color-red)',    icon: '⚡' },
};

const STATUS_META = {
  ok:          { label: 'Optimal',       color: 'var(--color-green)' },
  low:         { label: 'Düşük',         color: 'var(--color-amber)' },
  critical:    { label: 'Kritik',        color: 'var(--color-red)'   },
  unknown:     { label: 'Bilinmiyor',    color: 'var(--text-muted)'  },
  unavailable: { label: 'Kullanılamaz', color: 'var(--text-muted)'  },
  ready:        { label: 'Hazır',        color: 'var(--color-green)' },
  insufficient: { label: 'Yetersiz RAM', color: 'var(--color-red)'  },
  already_exists: { label: 'Mevcut',    color: 'var(--color-green)' },
  running:     { label: 'Çalışıyor',    color: 'var(--color-amber)' },
  stopped:     { label: 'Durduruldu',   color: 'var(--color-green)' },
};

function StatusBadge({ status }) {
  const meta = STATUS_META[status] || STATUS_META.unknown;
  return (
    <span className="opt-badge" style={{ color: meta.color, borderColor: meta.color }}>
      {meta.label}
    </span>
  );
}

function ScoreRing({ score }) {
  const color =
    score >= 80 ? 'var(--color-green)' :
    score >= 50 ? 'var(--color-amber)' : 'var(--color-red)';
  return (
    <div className="opt-score-ring" style={{ '--score-color': color }}>
      <svg viewBox="0 0 64 64" className="opt-score-svg">
        <circle cx="32" cy="32" r="28" className="opt-score-bg" />
        <circle
          cx="32" cy="32" r="28"
          className="opt-score-fg"
          style={{
            stroke: color,
            strokeDasharray: `${(score / 100) * 175.9} 175.9`,
          }}
        />
      </svg>
      <div className="opt-score-label" style={{ color }}>
        <span className="opt-score-num">{score}</span>
        <span className="opt-score-text">/ 100</span>
      </div>
    </div>
  );
}

function CopyButton({ command, copyKey }) {
  const { copiedCommand, copyCommand } = useOptimizerStore();
  const copied = copiedCommand === copyKey;

  return (
    <button
      className={`opt-copy-btn ${copied ? 'copied' : ''}`}
      onClick={() => copyCommand(command, copyKey)}
      title="Komutu Kopyala"
    >
      {copied ? '✓ Kopyalandı' : '📋 Kopyala'}
    </button>
  );
}

function CommandBlock({ command, copyKey, label }) {
  return (
    <div className="opt-command-block">
      {label && <p className="opt-command-label">{label}</p>}
      <div className="opt-command-row">
        <code className="opt-command-code">{command}</code>
        <CopyButton command={command} copyKey={copyKey} />
      </div>
    </div>
  );
}

// ── Tab: Pagefile ─────────────────────────────────────────────────────────────

function PagefileTab() {
  const { pagefile, fetchPagefile, isLoading } = useOptimizerStore();

  useEffect(() => { fetchPagefile(); }, []);

  if (isLoading && !pagefile) {
    return <div className="opt-loading">Pagefile analiz ediliyor...</div>;
  }

  if (!pagefile) return null;

  const statusMeta = STATUS_META[pagefile.status] || STATUS_META.unknown;

  return (
    <div className="opt-tab-content animate-fade-in">
      <div className="opt-section-header">
        <h3>🧊 Windows Sanal Bellek (Pagefile) Yöneticisi</h3>
        <p className="text-small">
          Pagefile'ı büyüterek RAM yetmediğinde modelin diske taşmasını kontrol altında tutun.
          NVMe SSD'de büyük pagefile, bellek baskısını önemli ölçüde azaltır.
        </p>
      </div>

      <div className="opt-cards-row">
        <div className="opt-info-card">
          <span className="opt-info-label">Mevcut Pagefile</span>
          <span className="opt-info-value">
            {pagefile.current_size_mb ? `${(pagefile.current_size_mb / 1024).toFixed(1)} GB` : 'Sistem Yönetimli'}
          </span>
          <StatusBadge status={pagefile.status} />
        </div>
        <div className="opt-info-card">
          <span className="opt-info-label">Fiziksel RAM</span>
          <span className="opt-info-value">{(pagefile.physical_ram_mb / 1024).toFixed(1)} GB</span>
        </div>
        <div className="opt-info-card">
          <span className="opt-info-label">Önerilen Boyut</span>
          <span className="opt-info-value" style={{ color: 'var(--accent-primary)' }}>
            {(pagefile.recommended_min_mb / 1024).toFixed(0)}–{(pagefile.recommended_max_mb / 1024).toFixed(0)} GB
          </span>
        </div>
      </div>

      <div className="opt-recommendation">
        <span className="opt-rec-icon">💡</span>
        <p>{pagefile.recommendation}</p>
      </div>

      {pagefile.status !== 'ok' && pagefile.powershell_command && (
        <>
          <div className="opt-warning-banner">
            ⚠️ Bu komutu <strong>Yönetici olarak açılmış PowerShell</strong>'de çalıştırın.
            Çalıştırmadan önce bilgisayarı yedeklemeniz önerilir.
          </div>
          <CommandBlock
            command={pagefile.powershell_command}
            copyKey="pagefile"
            label="Pagefile Ayarlama Komutu (Yönetici PowerShell):"
          />
        </>
      )}

      {pagefile.current_path && (
        <p className="text-small" style={{ marginTop: '0.5rem', opacity: 0.6 }}>
          📂 Mevcut konum: {pagefile.current_path}
        </p>
      )}
    </div>
  );
}

// ── Tab: Servisler ────────────────────────────────────────────────────────────

function ServicesTab() {
  const { services, processes, fetchServices, isLoading } = useOptimizerStore();
  const [activeSection, setActiveSection] = useState('services');

  useEffect(() => { fetchServices(); }, []);

  if (isLoading && services.length === 0) {
    return <div className="opt-loading">Servisler analiz ediliyor...</div>;
  }

  return (
    <div className="opt-tab-content animate-fade-in">
      <div className="opt-section-header">
        <h3>🛠️ Arka Plan Servis Optimizatörü</h3>
        <p className="text-small">
          Aşağıdaki servisleri durdurmak RAM boşaltır ve disk I/O hızını artırır.
          Komutlar doğrudan çalıştırılmaz — kopyalayıp <strong>Yönetici PowerShell</strong>'de çalıştırın.
        </p>
      </div>

      <div className="opt-sub-tabs">
        <button className={`opt-sub-tab ${activeSection === 'services' ? 'active' : ''}`}
          onClick={() => setActiveSection('services')}>
          Servisler ({services.length})
        </button>
        <button className={`opt-sub-tab ${activeSection === 'processes' ? 'active' : ''}`}
          onClick={() => setActiveSection('processes')}>
          İşlemler (Top {processes.length})
        </button>
      </div>

      {activeSection === 'services' && (
        <div className="opt-service-list">
          {services.map((svc) => {
            const risk = RISK_META[svc.risk] || RISK_META.safe;
            return (
              <div key={svc.name} className="opt-service-card">
                <div className="opt-service-header">
                  <div className="opt-service-info">
                    <span className="opt-service-name">{svc.display_name}</span>
                    <span className="opt-service-desc">{svc.description}</span>
                  </div>
                  <div className="opt-service-meta">
                    <StatusBadge status={svc.status} />
                    <span className="opt-risk-badge" style={{ color: risk.color }}>
                      {risk.icon} {risk.label}
                    </span>
                    {svc.memory_mb > 0 && (
                      <span className="opt-mem-badge">{svc.memory_mb.toFixed(0)} MB</span>
                    )}
                  </div>
                </div>
                {svc.status === 'running' && (
                  <div className="opt-service-commands">
                    <CommandBlock
                      command={svc.stop_command}
                      copyKey={`stop-${svc.name}`}
                      label="Durdur (oturum sonunda sıfırlanır):"
                    />
                    <CommandBlock
                      command={svc.disable_command}
                      copyKey={`disable-${svc.name}`}
                      label="Kalıcı Devre Dışı Bırak:"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {activeSection === 'processes' && (
        <div className="opt-process-list">
          <div className="opt-process-header-row">
            <span>İşlem Adı</span>
            <span>RAM</span>
            <span>CPU%</span>
            <span>Komut</span>
          </div>
          {processes.map((proc) => (
            <div key={proc.pid} className="opt-process-row">
              <span className="opt-proc-name" title={`PID: ${proc.pid}`}>{proc.name}</span>
              <span className="opt-proc-mem">{proc.memory_mb.toFixed(0)} MB</span>
              <span className="opt-proc-cpu">{proc.cpu_percent}%</span>
              <CopyButton command={proc.kill_command} copyKey={`kill-${proc.pid}`} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Tab: RAM Disk ─────────────────────────────────────────────────────────────

function RamDiskTab() {
  const { ramdisk, fetchRamdisk, isLoading } = useOptimizerStore();

  useEffect(() => { fetchRamdisk(); }, []);

  if (isLoading && !ramdisk) {
    return <div className="opt-loading">RAM Disk analiz ediliyor...</div>;
  }
  if (!ramdisk) return null;

  return (
    <div className="opt-tab-content animate-fade-in">
      <div className="opt-section-header">
        <h3>⚡ RAM Disk Yardımcısı</h3>
        <p className="text-small">
          Model dosyasını (.gguf) RAM'de oluşturulan sanal bir diske taşıyarak
          disk okuma hızını <strong>10-50x artırın</strong>. Bilgisayar kapandığında içerik silinir.
        </p>
      </div>

      <div className="opt-cards-row">
        <div className="opt-info-card">
          <span className="opt-info-label">Toplam RAM</span>
          <span className="opt-info-value">{(ramdisk.physical_ram_mb / 1024).toFixed(1)} GB</span>
        </div>
        <div className="opt-info-card">
          <span className="opt-info-label">Kullanılabilir</span>
          <span className="opt-info-value">{(ramdisk.available_ram_mb / 1024).toFixed(1)} GB</span>
        </div>
        <div className="opt-info-card">
          <span className="opt-info-label">Önerilen RAM Disk</span>
          <span className="opt-info-value" style={{ color: 'var(--accent-primary)' }}>
            {(ramdisk.recommended_ramdisk_mb / 1024).toFixed(1)} GB
          </span>
          <StatusBadge status={ramdisk.status} />
        </div>
      </div>

      {ramdisk.ramdisk_drives?.length > 0 && (
        <div className="opt-recommendation" style={{ borderColor: 'var(--color-green)' }}>
          <span className="opt-rec-icon">✓</span>
          <p>Mevcut RAM Disk sürücüleri tespit edildi: <strong>{ramdisk.ramdisk_drives.join(', ')}</strong></p>
        </div>
      )}

      {ramdisk.status === 'insufficient' && (
        <div className="opt-warning-banner">
          ⚠️ Mevcut kullanılabilir RAM ({(ramdisk.available_ram_mb / 1024).toFixed(1)} GB) RAM Disk için yeterli değil.
          Önce arka plan servislerini durdurun.
        </div>
      )}

      {ramdisk.status !== 'insufficient' && (
        <>
          <div className="opt-steps">
            <h4>Kurulum Adımları</h4>
            {ramdisk.setup_steps.map((step, i) => (
              <div key={i} className="opt-step">
                <span className="opt-step-num">{i + 1}</span>
                <span>{step}</span>
              </div>
            ))}
          </div>

          <CommandBlock
            command={ramdisk.powershell_command}
            copyKey="ramdisk"
            label={ramdisk.imdisk_installed ? "RAM Disk Oluşturma Komutu:" : "ImDisk Kurulduktan Sonra:"}
          />

          {!ramdisk.imdisk_installed && (
            <div className="opt-recommendation" style={{ marginTop: '0.75rem' }}>
              <span className="opt-rec-icon">📥</span>
              <p>
                ImDisk Toolkit henüz kurulu değil.{' '}
                <a
                  href="https://sourceforge.net/projects/imdisk-toolkit/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="opt-link"
                >
                  Buradan indirin →
                </a>
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Tab: Gelişmiş Donanım Hileleri (Hacks) ───────────────────────────────────

function HacksTab() {
  const {
    gpuProfile, fetchGpuProfile,
    triggerVramFlush, vramFlushResult,
    lockPriority, isPriorityLocked,
    applyWindowsPerformance, createLauncher, applyNvidiaTweak,
    isLoading
  } = useOptimizerStore();

  const [winPerfResult, setWinPerfResult] = useState(null);
  const [launcherResult, setLauncherResult] = useState(null);
  const [nvidiaResult, setNvidiaResult] = useState(null);

  useEffect(() => {
    fetchGpuProfile();
  }, []);

  const handleWinPerf = async () => {
    const res = await applyWindowsPerformance();
    setWinPerfResult(res);
  };

  const handleCreateLauncher = async () => {
    const res = await createLauncher();
    setLauncherResult(res);
  };

  const handleNvidiaTweak = async () => {
    const res = await applyNvidiaTweak();
    setNvidiaResult(res);
  };

  if (isLoading && !gpuProfile) {
    return <div className="opt-loading">GPU profilleri analiz ediliyor...</div>;
  }

  return (
    <div className="opt-tab-content animate-fade-in">
      <div className="opt-section-header">
        <h3>🚀 İleri Düzey Donanım Hileleri</h3>
        <p className="text-small">
          Mevcut ekran kartı ve işlemcinizden ekstra güç almak için yazılımsal bypass ve donanım limitleme hilelerini uygulayın.
        </p>
      </div>

      <div className="opt-hacks-grid">
        {/* Hack 1: VRAM Flush */}
        <div className="opt-hack-card">
          <div className="opt-hack-card-header">
            <h4>🧼 Windows VRAM Temizleyici (Flush VRAM)</h4>
            <span className="opt-badge-pill recommend">Önerilen</span>
          </div>
          <p className="text-small" style={{ margin: 'var(--space-2) 0' }}>
            Tauri arayüzü ve tarayıcıların VRAM önbelleğini temizler. Windows'u masaüstü bellek sayfalarını boşaltmaya zorlar.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginTop: 'auto' }}>
            <button className="btn btn-secondary btn-sm" onClick={triggerVramFlush}>
              🧹 Belleği Boşalt (Flush)
            </button>
            {vramFlushResult && (
              <span className="text-small" style={{ color: 'var(--color-green)' }}>
                ✓ {vramFlushResult.webview_processes_flushed || 0} süreç temizlendi!
              </span>
            )}
          </div>
        </div>

        {/* Hack 2: CPU Affinity & Priority Lock */}
        <div className="opt-hack-card">
          <div className="opt-hack-card-header">
            <h4>🧵 CPU Çekirdek Kilidi ve Yüksek Öncelik</h4>
            <span className="opt-badge-pill speed">Hız Artışı</span>
          </div>
          <p className="text-small" style={{ margin: 'var(--space-2) 0' }}>
            LLM sürecini sadece fiziksel CPU çekirdeklerine kilitler (Core Affinity) ve işlemci öncelik sınıfını Yüksek düzeye çıkarır.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginTop: 'auto' }}>
            <button
              className={`btn btn-secondary btn-sm ${isPriorityLocked ? 'active' : ''}`}
              onClick={lockPriority}
              disabled={isPriorityLocked}
            >
              🔒 Çekirdekleri Kilitle & Öncelik Yükselt
            </button>
            {isPriorityLocked && (
              <span className="text-small" style={{ color: 'var(--color-green)' }}>
                ✓ CPU & Süreç optimize edildi!
              </span>
            )}
          </div>
        </div>

        {/* Hack 5: Windows Performance Mode */}
        <div className="opt-hack-card">
          <div className="opt-hack-card-header">
            <h4>🖥️ Windows Görsel Efektlerini Kapat (VRAM Modu)</h4>
            <span className="opt-badge-pill recommend">Önerilen</span>
          </div>
          <p className="text-small" style={{ margin: 'var(--space-2) 0' }}>
            Windows'un pencerelerini en iyi performansa alarak pencerelerin ekran kartında işgal ettiği VRAM yükünü sıfırlar (~500MB VRAM).
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', marginTop: 'auto' }}>
            <button className="btn btn-secondary btn-sm" onClick={handleWinPerf}>
              ⚙️ Performans Modunu Uygula
            </button>
            {winPerfResult && (
              <span className="text-small" style={{ color: winPerfResult.status === 'ok' ? 'var(--color-green)' : 'var(--color-red)' }}>
                {winPerfResult.message}
              </span>
            )}
          </div>
        </div>

        {/* Hack 6: Create Sıfır VRAM Launcher */}
        <div className="opt-hack-card">
          <div className="opt-hack-card-header">
            <h4>🚀 Sıfır VRAM Başlatıcı Oluştur (.bat)</h4>
            <span className="opt-badge-pill speed">~400MB Tasarruf</span>
          </div>
          <p className="text-small" style={{ margin: 'var(--space-2) 0' }}>
            Proje klasörünüze, Tauri UI arayüzünün ekran kartını kullanmasını tamamen engelleyen bir başlatıcı script dosyası yazar.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', marginTop: 'auto' }}>
            <button className="btn btn-secondary btn-sm" onClick={handleCreateLauncher}>
              💾 Başlatıcı Script Üret
            </button>
            {launcherResult && (
              <span className="text-small" style={{ color: launcherResult.status === 'ok' ? 'var(--color-green)' : 'var(--color-red)' }}>
                {launcherResult.message}
              </span>
            )}
          </div>
        </div>

        {/* Hack 7: NVIDIA Sysmem Fallback Block */}
        <div className="opt-hack-card">
          <div className="opt-hack-card-header">
            <h4>🚫 NVIDIA RAM Taşmasını Engelle</h4>
            <span className="opt-badge-pill speed">Sistem Kilitlenmesi Önleyici</span>
          </div>
          <p className="text-small" style={{ margin: 'var(--space-2) 0' }}>
            VRAM dolduğunda sürücünün yavaş RAM'i kullanıp sistemi yavaşlatmasını engeller, CUDA'yı yüksek hızlı VRAM'de çalışmaya zorlar.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)', marginTop: 'auto' }}>
            <button className="btn btn-secondary btn-sm" onClick={handleNvidiaTweak}>
              🛡️ RAM Taşmasını Engelle (Tweak)
            </button>
            {nvidiaResult && (
              <div style={{ marginTop: 'var(--space-1)' }}>
                <span className="text-small" style={{ color: nvidiaResult.status === 'ok' ? 'var(--color-green)' : 'var(--color-red)', display: 'block', marginBottom: 'var(--space-2)' }}>
                  {nvidiaResult.message}
                </span>
                {nvidiaResult.status === 'admin_required' && nvidiaResult.powershell_command && (
                  <CommandBlock
                    command={nvidiaResult.powershell_command}
                    copyKey="nvidia-powershell"
                    label="Yönetici Yetkisi İçin PowerShell Komutu:"
                  />
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Hack 3: Multi-GPU Splitter */}
      {gpuProfile && gpuProfile.is_multi_gpu && (
        <div className="opt-hack-card" style={{ width: '100%' }}>
          <h4>🔀 Çoklu GPU Birleştirici (Tensor Splitter)</h4>
          <p className="text-small" style={{ margin: 'var(--space-1) 0 var(--space-3)' }}>
            Sistemdeki dahili (iGPU) veya harici ikincil ekran kartlarını ana ekran kartınızla birleştirerek model katmanlarını paylaştırır.
          </p>
          <div className="opt-gpu-split-container">
            {gpuProfile.gpus.map((gpu, i) => (
              <div key={gpu.index} className="opt-gpu-split-bar-item">
                <div className="opt-gpu-split-label">
                  <span>🎮 GPU {gpu.index}: {gpu.name}</span>
                  <span style={{ fontWeight: 'bold', color: 'var(--accent-primary)' }}>
                    %{(gpuProfile.tensor_split_recommended[i] * 100).toFixed(0)} Oran
                  </span>
                </div>
                <div className="opt-gpu-split-bar-bg">
                  <div
                    className="opt-gpu-split-bar-fg"
                    style={{ width: `${gpuProfile.tensor_split_recommended[i] * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="opt-recommendation" style={{ marginTop: 'var(--space-3)' }}>
            <span className="opt-rec-icon">⚙️</span>
            <p>
              Modelleri yüklerken çıkarım motoru bu oranları otomatik olarak kullanır. Donanım gücünü birleştirmek VRAM sınırınızı artırır.
            </p>
          </div>
        </div>
      )}

      {/* Hack 4: nvidia-smi power limiting */}
      {gpuProfile && gpuProfile.powershell_optimization_commands?.length > 0 && (
        <div className="opt-steps">
          <h4>❄️ NVIDIA Termal Throttling & Güç Sınırı Yöneticisi</h4>
          <p className="text-small" style={{ marginBottom: 'var(--space-3)' }}>
            LLM çıkarımı bellek bant genişliği limitlidir, yüksek çekirdek hızı kartı gereksiz ısıtır. Komutları kopyalayarak <strong>Yönetici PowerShell</strong>'de çalıştırın.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
            {gpuProfile.powershell_optimization_commands.map((cmd, idx) => (
              <CommandBlock
                key={idx}
                command={cmd}
                copyKey={`gpu-cmd-${idx}`}
                label={`Optimizasyon Adımı ${idx + 1}:`}
              />
            ))}
          </div>
        </div>
      )}

      {gpuProfile?.notes?.length > 0 && (
        <div className="opt-notes">
          {gpuProfile.notes.map((note, i) => (
            <p key={i} className="text-small" style={{ margin: '2px 0', opacity: 0.7 }}>
              • {note}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function SystemOptimizer({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('pagefile');
  const { status, fetchStatus } = useOptimizerStore();

  useEffect(() => {
    if (isOpen) fetchStatus();
  }, [isOpen]);

  if (!isOpen) return null;

  const tabs = [
    { id: 'pagefile', label: 'Sanal Bellek', icon: '🧊' },
    { id: 'services', label: 'Servisler',    icon: '🛠️' },
    { id: 'ramdisk',  label: 'RAM Disk',     icon: '⚡' },
    { id: 'hacks',    label: 'Gelişmiş (Hacks)', icon: '🚀' },
  ];

  return (
    <div className="opt-overlay" onClick={onClose}>
      <div className="opt-modal animate-scale-in" onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className="opt-modal-header">
          <div className="opt-header-left">
            <h2>🔧 Sistem Optimizasyonu</h2>
            <p className="text-small">Donanım yükseltmeden maksimum performans</p>
          </div>
          <div className="opt-header-right">
            {status && <ScoreRing score={status.optimization_score} />}
            <button className="btn btn-ghost btn-icon" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Score Recommendations */}
        {status?.recommendations?.length > 0 && (
          <div className="opt-recs">
            {status.recommendations.map((rec, i) => (
              <div key={i} className="opt-rec-item">
                <span>💡</span>
                <span>{rec}</span>
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div className="opt-tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`opt-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="opt-content">
          {activeTab === 'pagefile' && <PagefileTab />}
          {activeTab === 'services' && <ServicesTab />}
          {activeTab === 'ramdisk'  && <RamDiskTab />}
          {activeTab === 'hacks'    && <HacksTab />}
        </div>

        {/* Footer */}
        <div className="opt-footer">
          <span className="text-small" style={{ opacity: 0.5 }}>
            ⚠️ Komutlar uygulamada çalıştırılmaz — manuel olarak çalıştırmanız gerekir.
          </span>
          <button className="btn btn-primary btn-sm" onClick={onClose}>Kapat</button>
        </div>
      </div>
    </div>
  );
}
