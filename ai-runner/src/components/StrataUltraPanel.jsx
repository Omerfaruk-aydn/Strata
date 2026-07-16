import { useEffect, useState } from 'react';
import useExtremeStore from '../store/useExtremeStore';
import useModelStore from '../store/useModelStore';

function mb(bytes = 0) {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(2)} MB` : `${bytes} B`;
}

export default function StrataUltraPanel({ extreme }) {
  const { localModels, fetchLocalModels } = useModelStore();
  const [modelId, setModelId] = useState('');
  const [codec, setCodec] = useState('ternary-q05');
  const [groupSize, setGroupSize] = useState(128);
  const [threshold, setThreshold] = useState(0.125);
  const [conversion, setConversion] = useState(null);
  const [valueCount, setValueCount] = useState(16384);

  useEffect(() => { fetchLocalModels(); }, [fetchLocalModels]);
  useEffect(() => {
    if (!modelId && localModels.length) setModelId(localModels[0].id);
  }, [localModels, modelId]);

  const runConversion = async () => {
    if (!modelId) return;
    setConversion(await extreme.convertToStrata(modelId, null, groupSize, codec, threshold));
  };

  return <div className="extreme-ultra-layout">
    <section className="extreme-ultra-hero">
      <div><span className="extreme-card-kicker">EXPERIMENTAL LOW-BIT RUNTIME</span><h3>Strata Ultra</h3><p>GGUF modellerini bağımsız Strata formatına dönüştür, düşük-bit cache ve paging davranışını ölç.</p></div>
      <div className="ultra-status-pill"><i />{extreme.ultraCapabilities?.experimental ? 'DENEYSEL / AKTİF' : 'KAPALI'}</div>
    </section>
    <div className="extreme-ultra-grid">
      <section className="extreme-tool-card">
        <span className="extreme-card-kicker">FORMAT CONVERTER</span><h3>GGUF → Strata</h3>
        <label className="extreme-field"><span>Yerel GGUF kaynak</span><select value={modelId} onChange={(event) => setModelId(event.target.value)}>{!localModels.length && <option value="">GGUF model bulunamadı</option>}{localModels.map((model) => <option key={model.id} value={model.id}>{model.display_name} · {model.downloaded_quant}</option>)}</select></label>
        <div className="extreme-inline-fields"><label className="extreme-field"><span>Codec</span><select value={codec} onChange={(event) => setCodec(event.target.value)}><option value="ternary-q05">ternary-q05</option><option value="sparse05">sparse05</option></select></label><label className="extreme-field"><span>Grup</span><select value={groupSize} onChange={(event) => setGroupSize(Number(event.target.value))}><option value="64">64</option><option value="128">128</option><option value="256">256</option></select></label></div>
        {codec === 'sparse05' && <label className="extreme-field"><span>Sparse threshold</span><input type="number" min="0" max="10" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>}
        <button className="btn btn-primary" disabled={!modelId} onClick={runConversion}>Dönüştürmeyi başlat</button>
        {conversion && <div className="ultra-result">Hazır: {conversion.target || conversion.target_name}<br />MSE {conversion.quality?.mse ?? '—'} · Cosine {conversion.quality?.cosine_similarity ?? '—'}</div>}
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
