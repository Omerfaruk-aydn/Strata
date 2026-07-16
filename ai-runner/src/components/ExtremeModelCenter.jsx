import { useEffect, useMemo, useState } from 'react';
import useExtremeStore from '../store/useExtremeStore';
import useModelStore from '../store/useModelStore';
import useSettingsStore from '../store/useSettingsStore';
import './ExtremeModelCenter.css';

const STATUS_STYLES = {
  ideal: { icon: '◆', label: 'GPU’ya tam sığıyor', tone: 'success' },
  ready: { icon: '●', label: 'GPU + RAM ile hazır', tone: 'ready' },
  constrained: { icon: '▲', label: 'Bellek baskısıyla çalışabilir', tone: 'warning' },
  blocked: { icon: '■', label: 'Bu yapılandırma engellendi', tone: 'danger' },
};

const PRESET_LABELS = {
  safe: 'Güvenli',
  balanced: 'Dengeli',
  performance: 'Performans',
  maximum_capacity: 'Maksimum Kapasite',
};

function formatMb(value = 0) {
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`;
  return `${Math.round(value)} MB`;
}

function formatTime(milliseconds = 0) {
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toFixed(1)} sn`;
  return `${Math.round(milliseconds)} ms`;
}

export default function ExtremeModelCenter({ isOpen, onClose }) {
  const extreme = useExtremeStore();
  const settings = useSettingsStore();
  const { localModels, activeModel, loadModel, fetchLocalModels, loadReport } = useModelStore();
  const [tab, setTab] = useState('planner');
  const [sourceMode, setSourceMode] = useState('local');
  const [selectedKey, setSelectedKey] = useState('');
  const [preset, setPreset] = useState(settings.extremePreset || 'maximum_capacity');
  const [contextLength, setContextLength] = useState(settings.maxContextLength || 2048);
  const [parameterB, setParameterB] = useState(100);
  const [simulationQuant, setSimulationQuant] = useState('Q3_K_M');
  const [targetQuant, setTargetQuant] = useState('Q3_K_M');
  const [allowRequantize, setAllowRequantize] = useState(false);
  const [loadSuccess, setLoadSuccess] = useState(false);
  const [ultraCodec, setUltraCodec] = useState('ternary-q05');
  const [ultraGroupSize, setUltraGroupSize] = useState(128);
  const [ultraValueCount, setUltraValueCount] = useState(16384);
  const [ultraConversion, setUltraConversion] = useState(null);

  const modelOptions = useMemo(() => localModels.map((model, index) => ({
    key: `${index}:${model.id}:${model.downloaded_quant || ''}:${model.local_path || ''}`,
    model,
  })), [localModels]);
  const selectedModel = modelOptions.find((entry) => entry.key === selectedKey)?.model
    || extreme.ultraModels.find((model) => model.id === selectedKey)
    || null;
  const report = extreme.report;
  const status = STATUS_STYLES[report?.status] || STATUS_STYLES.blocked;
  const activeJobs = extreme.quantization.jobs?.filter((job) => ['queued', 'running'].includes(job.status)) || [];

  useEffect(() => {
    if (!isOpen) return undefined;
    Promise.all([
      extreme.fetchCapabilities(),
      extreme.fetchPresets(),
      extreme.fetchProfiles(),
      extreme.fetchQuantization(),
      extreme.fetchUltraCapabilities(),
      extreme.fetchUltraModels(),
      fetchLocalModels(),
    ]);
    return undefined;
  }, [isOpen]);

  useEffect(() => {
    if (!selectedKey && modelOptions.length > 0) setSelectedKey(modelOptions[0].key);
    if (selectedKey && !modelOptions.some((entry) => entry.key === selectedKey) && !extreme.ultraModels.some((model) => model.id === selectedKey)) {
      setSelectedKey(modelOptions[0]?.key || '');
    }
  }, [modelOptions, selectedKey, extreme.ultraModels]);

  useEffect(() => {
    if (!isOpen || activeJobs.length === 0) return undefined;
    const timer = setInterval(async () => {
      const state = await useExtremeStore.getState().fetchQuantization();
      if (!state?.jobs?.some((job) => ['queued', 'running'].includes(job.status))) {
        fetchLocalModels();
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [isOpen, activeJobs.length, fetchLocalModels]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const analyze = async () => {
    setLoadSuccess(false);
    if (sourceMode === 'local') {
      if (!selectedModel) return;
      await extreme.analyzeLocal(selectedModel.id, {
        quant: selectedModel.downloaded_quant,
        preset,
        contextLength,
        selectedGpuIndex: settings.selectedGpuIndex,
        tensorSplit: settings.tensorSplit,
      });
    } else {
      await extreme.simulate({
        parameterB,
        quant: simulationQuant,
        preset,
        contextLength,
        selectedGpuIndex: settings.selectedGpuIndex,
      });
    }
  };

  const loadRecommended = async () => {
    if (!selectedModel || !report || report.status === 'blocked') return;
    const runtime = report.runtime;
    setLoadSuccess(false);
    try {
      await loadModel(selectedModel.id, {
        quant: selectedModel.downloaded_quant,
        nGpuLayers: runtime.n_gpu_layers,
        contextLength: runtime.context_length,
        nThreads: runtime.n_threads,
        nBatch: runtime.n_batch,
        useMmap: runtime.use_mmap,
        useMlock: runtime.use_mlock,
        kvCacheType: runtime.kv_cache_type,
        flashAttn: runtime.flash_attn,
        speculativeDecoding: runtime.speculative_decoding,
        selectedGpuIndex: runtime.selected_gpu_index,
        tensorSplit: runtime.tensor_split,
        contextCompactionMode: settings.contextCompactionMode,
        extremePreset: preset,
        adaptiveLoad: true,
        adaptiveMaxAttempts: runtime.max_load_attempts,
        backendPreference: runtime.backend === 'unknown' ? 'auto' : runtime.backend,
      });
      setLoadSuccess(true);
      await extreme.fetchProfiles(selectedModel.id);
    } catch {
      // The model store already exposes the actionable backend error.
      setLoadSuccess(false);
    }
  };

  const startQuantization = async () => {
    if (!selectedModel) return;
    await extreme.startQuantization(
      selectedModel.id,
      selectedModel.downloaded_quant,
      targetQuant,
      allowRequantize,
    );
  };

  return (
    <div className="extreme-overlay" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="extreme-modal animate-scale-in" role="dialog" aria-modal="true" aria-labelledby="extreme-title">
        <header className="extreme-header">
          <div>
            <span className="extreme-eyebrow">AI RUNNER / CAPACITY ENGINE</span>
            <h2 id="extreme-title">Extreme Model Center</h2>
            <p>Çok büyük GGUF modellerini mevcut VRAM, RAM ve backend sınırları içinde planla ve doğrula.</p>
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose} aria-label="Kapat">✕</button>
        </header>

        <nav className="extreme-tabs" aria-label="Extreme Model araçları">
          {[
            ['planner', 'Kapasite Planı'],
            ['runtime', 'Çalışma Zamanı'],
            ['quantization', 'Quantization'],
            ['profiles', 'Donanım Profilleri'],
            ['ultra', 'Strata Ultra'],
          ].map(([id, label]) => (
            <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>

        <div className="extreme-content">
          {tab === 'planner' && (
            <div className="extreme-planner-grid">
              <aside className="extreme-control-panel">
                <div className="extreme-segmented">
                  <button className={sourceMode === 'local' ? 'active' : ''} onClick={() => setSourceMode('local')}>Yerel GGUF</button>
                  <button className={sourceMode === 'simulation' ? 'active' : ''} onClick={() => setSourceMode('simulation')}>İndirme Öncesi</button>
                </div>

                {sourceMode === 'local' ? (
                  <label className="extreme-field">
                    <span>Kaynak model</span>
                    <select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>
                      {modelOptions.length === 0 && <option value="">Yerel GGUF bulunamadı</option>}
                      {modelOptions.map(({ key, model }) => (
                        <option key={key} value={key}>{model.display_name} · {model.downloaded_quant}</option>
                      ))}
                    </select>
                  </label>
                ) : (
                  <div className="extreme-inline-fields">
                    <label className="extreme-field">
                      <span>Parametre</span>
                      <div className="extreme-input-suffix"><input type="number" min="1" max="10000" value={parameterB} onChange={(event) => setParameterB(Number(event.target.value))} /><b>B</b></div>
                    </label>
                    <label className="extreme-field">
                      <span>Quant</span>
                      <select value={simulationQuant} onChange={(event) => setSimulationQuant(event.target.value)}>
                        {['IQ1_S', 'IQ2_XXS', 'IQ2_XS', 'Q2_K', 'IQ3_XS', 'Q3_K_M', 'IQ4_XS', 'Q4_K_M', 'Q5_K_M'].map((quant) => <option key={quant}>{quant}</option>)}
                      </select>
                    </label>
                  </div>
                )}

                <label className="extreme-field">
                  <span>Kapasite profili</span>
                  <select value={preset} onChange={(event) => setPreset(event.target.value)}>
                    {(extreme.presets.length ? extreme.presets : Object.keys(PRESET_LABELS).map((name) => ({ name }))).map((item) => (
                      <option key={item.name} value={item.name}>{PRESET_LABELS[item.name] || item.name}</option>
                    ))}
                  </select>
                </label>

                <label className="extreme-field">
                  <span>Hedef context</span>
                  <div className="extreme-input-suffix"><input type="number" min="512" max="1048576" step="512" value={contextLength} onChange={(event) => setContextLength(Number(event.target.value))} /><b>tok</b></div>
                </label>

                <button className="btn btn-primary extreme-analyze" onClick={analyze} disabled={extreme.isLoading || (sourceMode === 'local' && !selectedModel)}>
                  {extreme.isLoading ? 'Donanım analiz ediliyor…' : 'Kapasiteyi analiz et'}
                </button>

                <div className="extreme-runtime-chip">
                  <span className={`runtime-dot ${extreme.capabilities?.gpu_offload_supported ? 'online' : ''}`} />
                  <div>
                    <b>{extreme.capabilities?.active_backend?.toUpperCase() || 'BACKEND'}</b>
                    <small>{extreme.capabilities?.gpu_offload_supported ? 'GPU offload kullanılabilir' : 'CPU / runtime kontrolü gerekli'}</small>
                  </div>
                </div>
              </aside>

              <main className="extreme-report-panel">
                {!report ? (
                  <div className="extreme-empty">
                    <div className="extreme-empty-mark">100B</div>
                    <h3>Önce çalışabilirliği ölç</h3>
                    <p>Planlayıcı model ağırlıklarını, KV cache’i, compute buffer’ı ve güvenlik paylarını ayrı hesaplar.</p>
                  </div>
                ) : (
                  <>
                    <div className={`extreme-status extreme-status-${status.tone}`}>
                      <span>{status.icon}</span>
                      <div><small>SONUÇ</small><strong>{status.label}</strong></div>
                      <b>{Math.round(report.gpu_layer_ratio * 100)}% GPU katmanı</b>
                    </div>

                    <div className="extreme-metrics">
                      <Metric label="Model ağırlığı" value={formatMb(report.memory.model_weights_mb)} />
                      <Metric label="Tahmini VRAM" value={formatMb(report.memory.estimated_vram_usage_mb)} />
                      <Metric label="RAM çalışma seti" value={formatMb(report.memory.estimated_ram_working_set_mb)} />
                      <Metric label="Tahmini hız" value={`${report.estimated_tokens_per_second_min}–${report.estimated_tokens_per_second_max} tok/sn`} />
                    </div>

                    <div className="extreme-memory-map">
                      <div className="extreme-section-heading"><span>BELLEK DAĞILIMI</span><b>{report.runtime.n_gpu_layers + report.runtime.cpu_layers} katman</b></div>
                      <MemoryRow label="GPU ağırlıkları" value={report.memory.gpu_weights_mb} total={report.memory.model_weights_mb} color="gpu" />
                      <MemoryRow label="CPU / RAM ağırlıkları" value={report.memory.cpu_weights_mb} total={report.memory.model_weights_mb} color="ram" />
                      <MemoryRow label="KV cache" value={report.memory.kv_cache_mb} total={Math.max(report.memory.model_weights_mb, report.memory.kv_cache_mb)} color="kv" />
                      <MemoryRow label="Compute buffer" value={report.memory.compute_buffer_mb} total={Math.max(report.memory.model_weights_mb, report.memory.compute_buffer_mb)} color="compute" />
                    </div>

                    <div className="extreme-config-grid">
                      <Config label="GPU katmanı" value={report.runtime.n_gpu_layers} />
                      <Config label="CPU katmanı" value={report.runtime.cpu_layers} />
                      <Config label="Context" value={report.runtime.context_length} />
                      <Config label="Batch" value={report.runtime.n_batch} />
                      <Config label="KV cache" value={report.runtime.kv_cache_type.toUpperCase()} />
                      <Config label="Depolama" value={report.memory.storage_mode.replaceAll('_', ' ')} />
                    </div>

                    {(report.blockers.length > 0 || report.warnings.length > 0 || report.actions.length > 0) && (
                      <div className="extreme-findings">
                        {report.blockers.map((item) => <p className="finding danger" key={item}><span>×</span>{item}</p>)}
                        {report.warnings.map((item) => <p className="finding warning" key={item}><span>!</span>{item}</p>)}
                        {report.actions.map((item) => <p className="finding info" key={item}><span>→</span>{item}</p>)}
                      </div>
                    )}

                    <div className="extreme-report-actions">
                      <span>{report.model.metadata_source === 'gguf' ? 'Gerçek GGUF metadata kullanıldı' : 'İndirme öncesi tahmin'}</span>
                      {sourceMode === 'local' && (
                        <button className="btn btn-primary" disabled={report.status === 'blocked' || useModelStore.getState().loadingModelId} onClick={loadRecommended}>
                          Önerilen planla yükle
                        </button>
                      )}
                    </div>
                    {loadSuccess && <div className="extreme-success">Model başarıyla yüklendi. {loadReport?.recovered_from_oom ? 'OOM fallback ile güvenli yapılandırma bulundu.' : 'İlk plan doğrulandı.'}</div>}
                  </>
                )}
              </main>
            </div>
          )}

          {tab === 'runtime' && (
            <div className="extreme-tools-layout">
              <section className="extreme-tool-card">
                <span className="extreme-card-kicker">MEASURED PERFORMANCE</span>
                <h3>Gerçek model benchmark’ı</h3>
                <p>Tahmin yerine yüklü model üzerinde token hızı, ilk token süresi ve bellek zirvesini ölçer.</p>
                <button className="btn btn-primary" onClick={() => extreme.runBenchmark(32)} disabled={!activeModel || extreme.isBenchmarking}>
                  {extreme.isBenchmarking ? 'Benchmark çalışıyor…' : '32 token benchmark çalıştır'}
                </button>
                {extreme.benchmark && (
                  <div className="benchmark-results">
                    <Metric label="Üretim hızı" value={`${extreme.benchmark.tokens_per_second} tok/sn`} />
                    <Metric label="İlk token" value={formatTime(extreme.benchmark.ttft_ms)} />
                    <Metric label="Toplam süre" value={formatTime(extreme.benchmark.total_time_ms)} />
                    <Metric label="RAM zirvesi" value={formatMb(extreme.benchmark.process_ram_peak_mb)} />
                  </div>
                )}
              </section>

              <section className="extreme-tool-card">
                <span className="extreme-card-kicker">ADAPTIVE OFFLOAD</span>
                <h3>Katmanları yeniden dengele</h3>
                <p>Boş VRAM değiştiğinde üretimler arasında modeli güvenli biçimde yeniden yükler. Aktif token üretimi sırasında bellek taşınmaz.</p>
                <button className="btn btn-secondary" onClick={() => extreme.rebalance(preset)} disabled={!activeModel || extreme.isLoading}>
                  Güncel donanıma göre yeniden dengele
                </button>
              </section>

              <section className="extreme-tool-card extreme-backend-card">
                <span className="extreme-card-kicker">NATIVE BACKEND</span>
                <h3>Kurulu çalışma motoru</h3>
                <div className="backend-list">
                  {(extreme.capabilities?.backends || []).map((backend) => (
                    <div className={`backend-row ${backend.active ? 'active' : ''}`} key={backend.name}>
                      <span>{backend.name.toUpperCase()}</span>
                      <b>{backend.active ? 'AKTİF' : backend.available ? 'HAZIR' : 'YÜKLÜ DEĞİL'}</b>
                      <small>{backend.reason}</small>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}

          {tab === 'quantization' && (
            <div className="extreme-tools-layout">
              <section className="extreme-tool-card quant-card">
                <span className="extreme-card-kicker">MODEL COMPRESSION</span>
                <h3>llama.cpp quantization</h3>
                <p>Yerel GGUF modelinden daha küçük bir quant oluşturur. Quantized bir modeli yeniden quantize etmek kaliteyi ek olarak düşürür.</p>
                <label className="extreme-field"><span>Kaynak model</span><select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>{modelOptions.map(({ key, model }) => <option key={key} value={key}>{model.display_name} · {model.downloaded_quant}</option>)}</select></label>
                <label className="extreme-field"><span>Hedef quant</span><select value={targetQuant} onChange={(event) => setTargetQuant(event.target.value)}>{(extreme.quantization.supported_quants || []).map((quant) => <option key={quant}>{quant}</option>)}</select></label>
                <label className="extreme-checkbox"><input type="checkbox" checked={allowRequantize} onChange={(event) => setAllowRequantize(event.target.checked)} /><span>Quantized kaynaktan yeniden quantize etmeye izin ver</span></label>
                {!extreme.quantization.available && <div className="extreme-inline-warning">llama-quantize bulunamadı. Araç yolu backend ortamında yapılandırılmalıdır.</div>}
                <button className="btn btn-primary" onClick={startQuantization} disabled={!selectedModel || !targetQuant || !extreme.quantization.available || extreme.isQuantizing || activeJobs.length > 0}>Quantization başlat</button>
              </section>

              <section className="extreme-tool-card jobs-card">
                <span className="extreme-card-kicker">JOB QUEUE</span>
                <h3>İşlem geçmişi</h3>
                {(extreme.quantization.jobs || []).length === 0 ? <p>Henüz quantization işlemi yok.</p> : (extreme.quantization.jobs || []).map((job) => (
                  <div className="quant-job" key={job.id}>
                    <div><b>{job.output_quant}</b><span>{job.status.toUpperCase()}</span></div>
                    <small>{job.message}</small>
                    <div className="quant-progress"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
                    {['queued', 'running'].includes(job.status) && <button className="btn btn-danger btn-sm" onClick={() => extreme.cancelQuantization(job.id)}>İptal et</button>}
                    {job.error && <p className="quant-error">{job.error}</p>}
                  </div>
                ))}
              </section>
            </div>
          )}

          {tab === 'ultra' && <UltraPanel extreme={extreme} modelOptions={modelOptions} selectedModel={selectedModel} selectedKey={selectedKey} setSelectedKey={setSelectedKey} groupSize={ultraGroupSize} setGroupSize={setUltraGroupSize} valueCount={ultraValueCount} setValueCount={setUltraValueCount} codec={ultraCodec} setCodec={setUltraCodec} conversion={ultraConversion} setConversion={setUltraConversion} />}

          {tab === 'profiles' && (
            <div className="profile-table-wrap">
              <div className="profile-intro"><div><span className="extreme-card-kicker">KNOWN-GOOD CONFIGURATIONS</span><h3>Donanıma özel çalışma profilleri</h3></div><button className="btn btn-secondary" onClick={() => extreme.fetchProfiles()}>Yenile</button></div>
              {extreme.profiles.length === 0 ? <div className="extreme-empty compact"><h3>Henüz doğrulanmış profil yok</h3><p>Bir modeli Extreme planıyla başarıyla yüklediğinde profil otomatik kaydedilir.</p></div> : (
                <div className="profile-table">
                  <div className="profile-row profile-head"><span>Model</span><span>Backend</span><span>Preset</span><span>GPU / Context</span><span>Benchmark</span></div>
                  {extreme.profiles.map((profile) => (
                    <div className="profile-row" key={profile.id}>
                      <span title={profile.model_id}>{profile.model_id}</span>
                      <span>{profile.backend.toUpperCase()}</span>
                      <span>{PRESET_LABELS[profile.preset] || profile.preset}</span>
                      <span>{profile.config.n_gpu_layers} katman / {profile.config.context_length}</span>
                      <span>{profile.benchmark ? `${profile.benchmark.tokens_per_second} tok/sn` : 'Ölçülmedi'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {extreme.error && <div className="extreme-error"><span>!</span><p>{extreme.error}</p><button onClick={extreme.clearError}>✕</button></div>}
        </div>
      </section>
    </div>
  );
}

function UltraPanel({ extreme, modelOptions, selectedModel, selectedKey, setSelectedKey, groupSize, setGroupSize, valueCount, setValueCount, codec, setCodec, conversion, setConversion }) {
  return <div className="extreme-ultra-layout">
    <section className="extreme-ultra-hero"><div><span className="extreme-card-kicker">EXPERIMENTAL LOW-BIT RUNTIME</span><h3>Strata Ultra</h3><p>Bağımsız `.strata` formatını, katman paging sistemini ve düşük-bit KV cache’i uygulama içinden ölç ve dene.</p></div><div className="ultra-status-pill"><i />{extreme.ultraCapabilities?.experimental ? 'DENEYSEL / AKTİF' : 'KAPALI'}</div></section>
    <div className="extreme-ultra-grid"><section className="extreme-tool-card"><span className="extreme-card-kicker">MEMORY LAB</span><h3>Bellek ve codec ölçümü</h3><div className="extreme-inline-fields"><label className="extreme-field"><span>Ağırlık sayısı</span><input type="number" min="128" value={valueCount} onChange={(event) => setValueCount(Number(event.target.value))} /></label><label className="extreme-field"><span>Grup boyutu</span><select value={groupSize} onChange={(event) => setGroupSize(Number(event.target.value))}><option value="32">32</option><option value="64">64</option><option value="128">128</option><option value="256">256</option></select></label></div><div className="ultra-actions"><button className="btn btn-primary" onClick={() => extreme.estimateUltraMemory(valueCount, groupSize)}>Belleği hesapla</button><button className="btn btn-secondary" onClick={() => extreme.runUltraBenchmark(valueCount, groupSize)}>Codec benchmark</button></div>{extreme.ultraMemoryReport && <div className="benchmark-results"><Metric label="FP16" value={formatMb(extreme.ultraMemoryReport.f16_bytes / 1048576)} /><Metric label="1-bit KV" value={formatMb(extreme.ultraMemoryReport.sign1_bytes / 1048576)} /><Metric label="Q0.5 KV" value={formatMb(extreme.ultraMemoryReport.ternary05_bytes / 1048576)} /><Metric label="Tasarruf" value={`${extreme.ultraMemoryReport.ternary05_saving_percent}%`} /></div>}</section><section className="extreme-tool-card"><span className="extreme-card-kicker">FORMAT CONVERTER</span><h3>GGUF → Strata</h3><p>Desteklenen yerel GGUF modelini deneysel düşük-bit konteynere dönüştür.</p><label className="extreme-field"><span>Kaynak model</span><select value={selectedKey} onChange={(event) => setSelectedKey(event.target.value)}>{extreme.ultraModels.length === 0 && <option value="">Model bulunamadı</option>}{extreme.ultraModels.map((model) => <option key={model.id} value={model.id}>{model.display_name || model.id}</option>)}</select></label><label className="extreme-field"><span>Hedef codec</span><select value={codec} onChange={(event) => setCodec(event.target.value)}><option value="ternary-q05">STRATA-Q0.5 · ternary</option><option value="sparse05">STRATA-Q0.5 · sparse05</option></select></label><button className="btn btn-primary" disabled={!selectedModel} onClick={async () => setConversion(await extreme.convertToStrata(selectedModel.id, null, groupSize, codec))}>Dönüştürmeyi başlat</button>{conversion && <p className="ultra-result">Hazır: {conversion.target_file || conversion.target_name}</p>}</section></div>
    <section className="extreme-ultra-capabilities"><span className="extreme-card-kicker">RUNTIME CAPABILITIES</span><div className="ultra-capability-list">{(extreme.ultraCapabilities?.features || []).map((feature) => <span key={feature}>✓ {feature}</span>)}</div><small>Deneysel CPU runtime; tokenizer, mimari eşleme ve GPU kernel kapsamı modele göre değişir.</small></section>
  </div>;
}

function Metric({ label, value }) {
  return <div className="extreme-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Config({ label, value }) {
  return <div className="extreme-config"><span>{label}</span><b>{value}</b></div>;
}

function MemoryRow({ label, value, total, color }) {
  const width = Math.max(value > 0 ? 2 : 0, Math.min(100, (value / Math.max(total, 1)) * 100));
  return (
    <div className="memory-row">
      <div><span>{label}</span><b>{formatMb(value)}</b></div>
      <div className="memory-track"><i className={`memory-${color}`} style={{ width: `${width}%` }} /></div>
    </div>
  );
}
