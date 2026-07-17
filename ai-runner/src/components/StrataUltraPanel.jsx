import { useEffect, useState } from 'react';
import useExtremeStore from '../store/useExtremeStore';
import useModelStore from '../store/useModelStore';
import { apiFetch } from '../api/client';

function mb(bytes = 0) {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : `${bytes} B`;
}

export default function StrataUltraPanel({ extreme }) {
  const { localModels, fetchLocalModels } = useModelStore();
  const ggufModels = localModels.filter((model) => model.runtime !== 'strata');
  const [modelId, setModelId] = useState('');
  const [codec, setCodec] = useState('ternary-q05');
  const [groupSize, setGroupSize] = useState(128);
  const [threshold, setThreshold] = useState(0.125);
  const [conversion, setConversion] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [valueCount, setValueCount] = useState(16384);
  const [strataFile, setStrataFile] = useState('');
  const [prompt, setPrompt] = useState('Merhaba, kendini kısaca tanıt.');
  const [generation, setGeneration] = useState(null);
  const [generating, setGenerating] = useState(false);
  const [generationError, setGenerationError] = useState('');
  const selectedStrataModel = (extreme.ultraModels || []).find((model) => model.file === strataFile);

  useEffect(() => { fetchLocalModels(); }, [fetchLocalModels]);
  useEffect(() => {
    if (!modelId && ggufModels.length) setModelId(ggufModels[0].id);
  }, [ggufModels, modelId]);
  useEffect(() => {
    if (!strataFile && extreme.ultraModels?.length) setStrataFile(extreme.ultraModels[0].file);
  }, [extreme.ultraModels, strataFile]);

  const runConversion = async () => {
    if (!modelId) return;
    const result = await extreme.convertToStrata(modelId, null, groupSize, codec, threshold);
    setConversion(result);
    const file = result?.target ? result.target.split(/[\\/]/).pop() : null;
    await Promise.all([
      extreme.fetchUltraModels(),
      fetchLocalModels(),
    ]);
    if (file) {
      setStrataFile(file);
    }
    if (file) setInspection(await extreme.inspectStrataModel(file));
  };

  const runGeneration = async () => {
    if (!strataFile || !prompt.trim()) return;
    setGenerating(true);
    setGenerationError('');
    try {
      const response = await apiFetch('/api/ultra/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_file: strataFile,
          embedding_tensor: 'token_embd.weight',
          output_tensor: 'output.weight',
          width: selectedStrataModel?.width || 896,
          context_capacity: 2048,
          kv_mode: 'ternary05',
          prompt: prompt.trim(),
          max_new_tokens: 8,
          timeout_s: 300,
          memory_budget_bytes: 536870912,
          resident_window: 2,
          backend: 'cuda',
        }),
      });
      setGeneration(await response.json());
    } catch (error) {
      setGenerationError(error.message || 'Strata üretimi başarısız oldu.');
    } finally {
      setGenerating(false);
    }
  };

  return <div className="extreme-ultra-layout">
    <section className="extreme-ultra-hero">
      <div><span className="extreme-card-kicker">EXPERIMENTAL LOW-BIT RUNTIME</span><h3>Strata Ultra</h3><p>GGUF modellerini bağımsız Strata formatına dönüştür, düşük-bit cache ve paging davranışını ölç.</p></div>
      <div className="ultra-status-pill"><i />{extreme.ultraCapabilities?.experimental ? 'DENEYSEL / AKTİF' : 'KAPALI'}</div>
    </section>
    <div className="extreme-ultra-grid">
      <section className="extreme-tool-card">
        <span className="extreme-card-kicker">STRATA GPU CHAT</span><h3>Hazır modeli kullan</h3>
        <label className="extreme-field"><span>Strata model</span><select value={strataFile} onChange={(event) => setStrataFile(event.target.value)}>{(extreme.ultraModels || []).map((model) => <option key={model.file} value={model.file}>{model.file}</option>)}</select></label>
        <label className="extreme-field"><span>Prompt</span><textarea rows="3" value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Modele bir şey sor..." /></label>
        <button className="btn btn-primary" disabled={!strataFile || !prompt.trim() || generating} onClick={runGeneration}>{generating ? 'GPU üzerinde üretiliyor…' : 'GPU ile cevap üret'}</button>
        {generationError && <div className="ultra-result">⚠ {generationError}</div>}
        {generation && <div className="ultra-result"><strong>Cevap</strong><br />{generation.text}<br /><small>{generation.generated_tokens} token · {generation.backend} · {generation.blocks} blok</small></div>}
      </section>
      <section className="extreme-tool-card">
        <span className="extreme-card-kicker">FORMAT CONVERTER</span><h3>GGUF → Strata</h3>
        <label className="extreme-field"><span>Yerel GGUF kaynak</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}>{!ggufModels.length && <option value="">GGUF model bulunamadı</option>}{ggufModels.map((model) => <option key={model.id} value={model.id}>{model.display_name} · {model.downloaded_quant}</option>)}</select></label>
        <div className="extreme-inline-fields"><label className="extreme-field"><span>Codec</span><select value={codec} onChange={(event) => setCodec(event.target.value)}><option value="ternary-q05">ternary-q05</option><option value="sparse05">sparse05</option></select></label><label className="extreme-field"><span>Grup</span><select value={groupSize} onChange={(event) => setGroupSize(Number(event.target.value))}><option value="64">64</option><option value="128">128</option><option value="256">256</option></select></label></div>
        {codec === 'sparse05' && <label className="extreme-field"><span>Sparse threshold</span><input type="number" min="0" max="10" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>}
        <button className="btn btn-primary" disabled={!modelId} onClick={runConversion}>Dönüştürmeyi başlat</button>
        {conversion && <div className="ultra-result">Hazır: {conversion.target || conversion.target_name}<br />MSE {conversion.quality?.mse ?? '—'} · Cosine {conversion.quality?.cosine_similarity ?? '—'}</div>}
        {inspection && <div className="ultra-result">Packed {mb(inspection.packed_bytes)} · Scales {mb(inspection.scales_bytes)}<br />Generation: {inspection.ready_for_experimental_generation ? 'hazır' : 'eksik mimari/tensor'}</div>}
      </section>
      <section className="extreme-tool-card">
        <span className="extreme-card-kicker">MEMORY LAB</span><h3>Codec benchmark</h3>
        <label className="extreme-field"><span>Değer sayısı</span><input type="number" min="128" value={valueCount} onChange={(event) => setValueCount(Number(event.target.value))} /></label>
        <button className="btn btn-secondary" onClick={() => extreme.runUltraBenchmark(valueCount, groupSize, threshold)}>Benchmark çalıştır</button>
        {extreme.ultraBenchmark && <div className="ultra-result">Ternary: {mb(extreme.ultraBenchmark.packed_bytes)} · Sparse: {mb(extreme.ultraBenchmark.sparse05?.packed_bytes)}<br />Sparse MSE: {extreme.ultraBenchmark.sparse05?.quality?.mse ?? '—'}</div>}
      </section>
    </div>
    <section className="extreme-ultra-capabilities"><span className="extreme-card-kicker">RUNTIME CAPABILITIES</span><div className="ultra-capability-list">{(extreme.ultraCapabilities?.features || []).map((feature) => <span key={feature}>✓ {feature}</span>)}</div><small>Deneysel CPU runtime; kalite ve sparsity değerleri model bazında benchmark edilmelidir.</small></section>
  </div>;
}
