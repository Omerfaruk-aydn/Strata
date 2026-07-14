# AI Runner — Kurumsal Düzey Teknik Spesifikasyon
### Yerel LLM Çalıştırma Platformu — Tam Ürün ve Mühendislik Promptu
**Versiyon:** 3.0 · **Doküman Tipi:** PRD + Teknik Spesifikasyon (kodlama ajanına doğrudan verilebilir)
**Bu sürümde yeni:** 20 maddelik genişletilmiş özellik kataloğu (Faz 7+), etki/efor önceliklendirme matrisi, güncellenmiş yol haritası.

---

## İçindekiler

1. Yönetici Özeti
2. Ürün Vizyonu ve Değer Önerisi
3. Rekabet Analizi
4. Kullanıcı Personaları
5. Kapsam Tanımı
6. Fonksiyonel Gereksinimler (ID'li)
7. Fonksiyonel Olmayan Gereksinimler
8. Sistem Mimarisi
9. Veri Modelleri ve Şemalar
10. API Spesifikasyonu
11. Donanım Profilleme ve Optimizasyon Algoritması
12. Quantization Karar Matrisi
13. Tasarım Sistemi (Genişletilmiş)
14. Bileşen Envanteri
15. Durum Yönetimi Mimarisi
16. Hata Yönetimi ve Dayanıklılık
17. Güvenlik Tehdit Modeli
18. Test Stratejisi
19. Gözlemlenebilirlik ve Loglama
20. Paketleme, Dağıtım, Güncelleme
21. Dizin Yapısı (Genişletilmiş)
22. Yol Haritası (Süre Tahminli)
23. Başarı Metrikleri
24. Risk Kayıt Defteri
25. Açık Sorular
26. Genişletilmiş Özellik Kataloğu (Faz 7+)
27. Öncelik / Karmaşıklık Matrisi (Faz 7+)
28. Sözlük

---

## 1. Yönetici Özeti

AI Runner, kısıtlı donanımda (4–12GB VRAM) büyük dil modellerini (7B–100B+ parametre) çalıştırmaya odaklanan, masaüstü tabanlı, donanım şeffaflığını ürünün merkezine koyan bir LLM çalıştırma platformudur. Ürün, mevcut araçların (LM Studio, Ollama, text-generation-webui) sunmadığı şu üç şeyi bir arada verir:

1. **Otomatik + görünür donanım optimizasyonu** — kullanıcı elle ayar yapmak zorunda kalmaz, ama isterse her kararı görüp değiştirebilir.
2. **Gerçek zamanlı katman telemetrisi** — modelin hangi parçasının nerede çalıştığını canlı izleme.
3. **Tek paket, düşük ayak izi** — Tauri tabanlı, Electron'a göre ~50 kat daha küçük dağıtım boyutu.

---

## 2. Ürün Vizyonu ve Değer Önerisi

**Vizyon cümlesi:** "Herkesin kendi bilgisayarında, donanımının sınırlarını gerçekten anlayarak, en büyük açık modelleri güvenle çalıştırabildiği araç."

**Değer önerisi katmanları:**

| Katman | Sunulan Değer |
|---|---|
| Fonksiyonel | Zayıf donanımda büyük modelleri çalıştırabilme |
| Duygusal | "Ne olduğunu anlıyorum" hissi — kara kutu değil, şeffaf sistem |
| Sosyal | Paylaşılabilir performans profilleri ("benim RTX 3060'ımda 70B model şu hızda çalışıyor") |

**Farklılaşma ekseni:** Rakiplerin çoğu "çalıştır ve unut" felsefesindedir. AI Runner "çalıştır ve *anla*" felsefesini benimser — bu, hem güven inşa eder hem de kullanıcıyı donanım yatırımı kararlarında bilgilendirir.

---

## 3. Rekabet Analizi

| Ürün | Güçlü Yönü | Zayıf Yönü | AI Runner Farkı |
|---|---|---|---|
| LM Studio | Olgun UI, model kataloğu | Kapalı kaynak, telemetri şeffaflığı yok | Açık mimari + canlı katman görselleştirme |
| Ollama | Basit CLI, güçlü model kütüphanesi | GUI zayıf, ileri ayar sınırlı | Zengin GUI + ileri kullanıcı kontrolü |
| text-generation-webui | Çok esnek, eklenti zengin | Karmaşık kurulum, dağınık UX | Tek tık kurulum, cilalı UX |
| GPT4All | Hafif, kolay kurulum | Büyük modellerde zayıf | 100B+ model hedefli offload mimarisi |

---

## 4. Kullanıcı Personaları

**Persona A — "Kısıtlı Geliştirici" (Deniz, 27)**
RTX 4060 (8GB) kullanıyor, yerel modelle kod tamamlama denemek istiyor. VRAM hatalarından bıkmış. İhtiyacı: "Bu model benim GPU'mda çalışır mı?" sorusuna kurulum yapmadan cevap.

**Persona B — "Meraklı Araştırmacı" (Elif, 34)**
Farklı quantization seviyelerinin çıktı kalitesini karşılaştırıyor. İhtiyacı: yan yana model karşılaştırma, ayrıntılı metrikler.

**Persona C — "Gizlilik Odaklı Kullanıcı" (Kaan, 41)**
Şirket verisini buluta göndermek istemiyor. İhtiyacı: %100 yerel çalışma garantisi, ağ trafiği denetimi.

---

## 5. Kapsam Tanımı

**Kapsam İçi (v1.0):**
- Tek makinede, GGUF formatlı modellerin yüklenmesi ve çalıştırılması
- Otomatik donanım profilleme ve offload planlama
- Sohbet arayüzü, model yönetimi, canlı telemetri
- Yerel OpenAI-uyumlu API sunucusu

**Kapsam Dışı (v1.0, gelecek fazlara ertelendi):**
- Dağıtık/çoklu makine inference (Petals benzeri) — Faz 6+
- Model fine-tuning / eğitim
- Mobil uygulama
- Bulut senkronizasyonu

---

## 6. Fonksiyonel Gereksinimler

### 6.1 Model Yönetimi
- **FR-101:** Kullanıcı, HuggingFace Hub'da GGUF formatlı modelleri arama çubuğundan arayabilmeli; sonuçlar model adı, boyutu, indirilme sayısı, lisans ile listelenmeli.
- **FR-102:** İndirme işlemi duraklatılabilir/devam ettirilebilir olmalı (chunk tabanlı, resumable).
- **FR-103:** Her model kartında donanım uyumluluk rozeti gösterilmeli: 🟢 Rahat çalışır / 🟡 Kısıtlı çalışır (kısmi offload) / 🔴 Bu donanımda önerilmez.
- **FR-104:** Kullanıcı bir modeli indirmeden önce, seçtiği quantization seviyesine göre tahmini VRAM/RAM/disk kullanımını görmeli.
- **FR-105:** Yerel model kütüphanesinde: model boyutu, son kullanım tarihi, toplam disk kullanımı, tek tıkla silme.
- **FR-106:** Model meta verisi önbelleğe alınmalı (offline modda da kütüphane görüntülenebilmeli).

### 6.2 Donanım Profilleme
- **FR-201:** Uygulama açılışında GPU modeli, VRAM (toplam/boşta), sistem RAM'i, disk tipi (SSD/HDD) ve boş disk alanı otomatik tespit edilmeli.
- **FR-202:** Çoklu GPU durumunda kullanıcı hangi GPU'nun kullanılacağını seçebilmeli.
- **FR-203:** Profil sonucu "Donanım Kartı" olarak gösterilmeli ve manuel yenilenebilmeli (harici GPU takılıp çıkarılması durumları için).
- **FR-204:** Donanım profili değişikliklerinde (örn. başka bir uygulama VRAM'i doldurduğunda) sistem uyarı vermeli.

### 6.3 Inference Motoru
- **FR-301:** Model yüklenirken `n_gpu_layers` değeri otomatik hesaplanmalı, kullanıcı isterse manuel override edebilmeli.
- **FR-302:** Üretim çıktısı token-token streaming olarak arayüze akmalı, gecikme <100ms/token hedeflenmeli (donanıma bağlı).
- **FR-303:** Kullanıcı üretimi herhangi bir anda durdurabilmeli (interrupt).
- **FR-304:** Context penceresi doldurulduğunda otomatik "sliding window" veya kullanıcı uyarısı ile özetleme seçeneği sunulmalı.
- **FR-305:** Speculative decoding opsiyonel olarak açılabilmeli (küçük taslak model seçimi ile).
- **FR-306:** Sistem promptu, sıcaklık (temperature), top-p, top-k, repeat-penalty gibi üretim parametreleri arayüzden ayarlanabilmeli.
- **FR-307:** Aynı anda yalnızca bir model aktif olabilir (v1.0 kapsamında); model değişimi güvenli şekilde önceki modeli bellekten boşaltmalı.

### 6.4 Kullanıcı Arayüzü
- **FR-401:** Üç panelli düzen: Model Rafı (sol, daraltılabilir), Sohbet Konsolu (orta), Telemetri Paneli (sağ, daraltılabilir).
- **FR-402:** Telemetri panelinde canlı: VRAM kullanımı (bar + MB), RAM kullanımı, disk I/O, token/saniye, ilk token gecikmesi (TTFT), GPU sıcaklığı (destekleniyorsa).
- **FR-403:** Katman dağılım çubuğu: modelin toplam katman sayısının yüzde kaçının GPU/RAM/disk'te olduğunu segmentli çubukla gösterme.
- **FR-404:** Sohbet geçmişi kalıcı olmalı (yerel veritabanında), oturumlar arasında gezinilebilmeli.
- **FR-405:** Sohbet dışa aktarma: Markdown ve JSON formatları.
- **FR-406:** Klavye kısayolları: yeni sohbet (Ctrl/Cmd+N), gönder (Enter), durdur (Esc), arama (Ctrl/Cmd+K).
- **FR-407:** Karanlık/aydınlık tema anahtarı (varsayılan karanlık, telemetri estetiği aydınlıkta da korunmalı).

### 6.5 API Katmanı
- **FR-501:** `/v1/chat/completions` OpenAI şemasıyla uyumlu, hem streaming hem non-streaming destekli.
- **FR-502:** `/v1/models` endpoint'i kurulu modelleri OpenAI formatında listelemeli (üçüncü parti araç uyumluluğu için).
- **FR-503:** API sunucusu varsayılan olarak yalnızca `127.0.0.1`'de dinlemeli; ağ genelinde erişim açık rıza gerektirmeli ve uyarı göstermeli.
- **FR-504:** Opsiyonel API anahtarı ile erişim kısıtlama.

### 6.6 Ayarlar ve Yapılandırma
- **FR-601:** Genel ayarlar: varsayılan model, varsayılan sistem promptu, tema, dil (TR/EN).
- **FR-602:** Depolama ayarları: model indirme klasörü konumu, önbellek boyutu sınırı.
- **FR-603:** İleri ayarlar: thread sayısı, mmap kullanımı, batch boyutu — varsayılan gizli, "İleri Mod" ile açılır.
- **FR-604:** Ayarlar JSON dosyasında saklanmalı, dışa/içe aktarılabilir (makine değişiminde taşınabilirlik).

### 6.7 Oturum / Sohbet Yönetimi
- **FR-701:** Çoklu sohbet sekmesi desteği.
- **FR-702:** Her sohbete özel model + parametre override imkânı.
- **FR-703:** Sohbet yeniden adlandırma, silme, sabitleme (pin).

### 6.8 Eklenti Sistemi (Faz 6+ için tasarım hazırlığı)
- **FR-801:** Eklenti manifestosu (JSON) ile üçüncü parti araç tanımlama arayüzü (v1.0'da yalnızca mimari iskelet, aktif değil).

---

## 7. Fonksiyonel Olmayan Gereksinimler

| Kategori | Gereksinim |
|---|---|
| Performans | Uygulama soğuk başlangıç <3sn; model yükleme ilerlemesi her zaman görünür |
| Güvenilirlik | Model yükleme hatası uygulamayı çökertmemeli, anlaşılır hata mesajı vermeli |
| Taşınabilirlik | Windows 10+, macOS 12+, Ubuntu 20.04+ desteklenmeli |
| Erişilebilirlik | Klavye ile tam gezinilebilirlik, görünür focus durumları, WCAG AA kontrast oranları |
| Yerelleştirme | Arayüz metinleri i18n dosyalarından okunmalı (TR/EN başlangıç) |
| Bakım Yapılabilirlik | Backend ve frontend bağımsız test edilebilir, %70+ birim test kapsamı hedefi (çekirdek modüllerde) |
| Gizlilik | Varsayılan olarak sıfır dış veri iletimi (model indirme hariç) |

---

## 8. Sistem Mimarisi

### 8.1 Bileşen Diyagramı

```
┌──────────────────────────── Tauri Shell (Rust) ────────────────────────────┐
│  • Pencere yönetimi  • Sidecar process orkestrasyonu  • OS entegrasyonu    │
│                                                                              │
│  ┌────────────────────── React Frontend (WebView) ────────────────────┐   │
│  │  ModelShelf.jsx │ ChatConsole.jsx │ TelemetryPanel.jsx │ Settings   │   │
│  │            state: Zustand store (sessions, models, telemetry)       │   │
│  └───────────────────────────────┬───────────────────────────────────┘   │
│                     HTTP (REST) + WebSocket (localhost:8420)              │
│  ┌────────────────────────────────▼───────────────────────────────────┐  │
│  │                  Python Backend (FastAPI, sidecar)                  │  │
│  │  ┌───────────────┐ ┌──────────────┐ ┌───────────────────────────┐ │  │
│  │  │ HardwareProfiler│ │ ModelManager │ │ InferenceEngine           │ │  │
│  │  │ (psutil,pynvml)│ │ (hf_hub)     │ │ (llama_cpp)               │ │  │
│  │  └───────┬───────┘ └──────┬───────┘ └─────────────┬─────────────┘ │  │
│  │          └────────────────┴───────────┬─────────────┘             │  │
│  │                              ┌─────────▼──────────┐                │  │
│  │                              │  MemoryManager      │                │  │
│  │                              │  (offload planlayıcı)│               │  │
│  │                              └──────────────────────┘               │  │
│  │  ┌──────────────────────────────────────────────────────────────┐ │  │
│  │  │  SQLite (sohbet geçmişi, ayarlar, model meta önbelleği)       │ │  │
│  │  └──────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Süreç Yaşam Döngüsü (Sequence — Model Yükleme)

```
Kullanıcı → Frontend: "Modeli Yükle" tıklar
Frontend → Backend: POST /api/models/{id}/load
Backend → HardwareProfiler: mevcut VRAM/RAM sorgula
Backend → MemoryManager: offload planı hesapla (n_gpu_layers, context)
Backend → InferenceEngine: model = Llama(path, n_gpu_layers=X, ...)
InferenceEngine → Backend: yükleme ilerlemesi (WebSocket ile stream)
Backend → Frontend: WS "model_loading_progress" event'leri
InferenceEngine → Backend: yükleme tamamlandı + katman dağılım raporu
Backend → Frontend: WS "model_ready" + telemetri anlık görüntüsü
```

### 8.3 Süreç İzolasyonu Gerekçesi
Python backend'i ayrı bir sidecar process olarak çalıştırılır (Tauri'nin `shell` API'si ile). Bu, bir model çöktüğünde (OOM vb.) ana arayüzün donmamasını sağlar — backend yeniden başlatılabilir, kullanıcı sohbet geçmişini kaybetmez (SQLite'ta kalıcı).

---

## 9. Veri Modelleri ve Şemalar

```json
// ModelMetadata
{
  "id": "TheBloke/Llama-3-70B-GGUF",
  "display_name": "Llama 3 70B",
  "parameter_count": 70000000000,
  "available_quants": ["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0"],
  "license": "llama3",
  "context_length": 8192,
  "downloaded_quant": "Q4_K_M",
  "file_size_bytes": 42500000000,
  "local_path": "/models/llama3-70b-q4km.gguf",
  "last_used": "2026-07-10T14:32:00Z"
}

// HardwareProfile
{
  "gpu": { "name": "RTX 4070", "vram_total_mb": 12288, "vram_free_mb": 10450 },
  "ram": { "total_mb": 32768, "free_mb": 18200 },
  "disk": { "type": "SSD", "free_gb": 240 },
  "cpu": { "cores": 12, "threads": 20 }
}

// OffloadPlan (MemoryManager çıktısı)
{
  "model_id": "TheBloke/Llama-3-70B-GGUF",
  "quant": "Q4_K_M",
  "total_layers": 80,
  "gpu_layers": 44,
  "cpu_layers": 30,
  "disk_streamed_layers": 6,
  "estimated_tokens_per_sec": 4.2,
  "context_length_recommended": 4096,
  "warnings": ["Disk streaming aktif — SSD önerilir, bulunan: SSD ✓"]
}

// TelemetrySnapshot (WebSocket ile periyodik yayınlanır)
{
  "timestamp": "2026-07-13T10:15:32Z",
  "vram_used_mb": 11200,
  "ram_used_mb": 9400,
  "tokens_per_sec": 4.1,
  "ttft_ms": 320,
  "active_layer_distribution": { "gpu": 44, "ram": 30, "disk": 6 }
}

// ChatSession
{
  "id": "sess_8f2a",
  "title": "Kod inceleme yardımı",
  "model_id": "TheBloke/Llama-3-70B-GGUF",
  "messages": [
    { "role": "user", "content": "...", "timestamp": "..." },
    { "role": "assistant", "content": "...", "timestamp": "...", "tokens_generated": 128 }
  ],
  "params": { "temperature": 0.7, "top_p": 0.9, "system_prompt": "..." }
}
```

---

## 10. API Spesifikasyonu

| Method | Endpoint | Açıklama |
|---|---|---|
| GET | `/api/hardware/profile` | Güncel donanım profilini döndürür |
| GET | `/api/models/search?q=` | HuggingFace model arama |
| POST | `/api/models/{id}/download` | İndirme başlatır (SSE ile ilerleme) |
| GET | `/api/models/local` | Kurulu modelleri listeler |
| DELETE | `/api/models/local/{id}` | Modeli diskten siler |
| POST | `/api/models/{id}/plan` | Belirli quant için offload planı hesaplar (indirmeden önce önizleme) |
| POST | `/api/models/{id}/load` | Modeli belleğe yükler |
| POST | `/api/models/unload` | Aktif modeli bellekten kaldırır |
| WS | `/ws/inference` | Token streaming + telemetri kanalı |
| POST | `/v1/chat/completions` | OpenAI-uyumlu sohbet uç noktası |
| GET | `/v1/models` | OpenAI-uyumlu model listesi |
| GET/PUT | `/api/settings` | Ayarları okur/günceller |
| GET/POST/DELETE | `/api/sessions` | Sohbet oturumu CRUD |

**Örnek İstek — Offload Planı Önizleme:**
```
POST /api/models/TheBloke%2FLlama-3-70B-GGUF/plan
{ "quant": "Q4_K_M" }

→ 200 OK
{ "gpu_layers": 44, "cpu_layers": 30, "disk_streamed_layers": 6,
  "estimated_tokens_per_sec": 4.2, "fits_comfortably": false,
  "recommendation": "Q4_K_M ile kısmi disk streaming gerekir. Q2_K seçilirse tamamı VRAM+RAM'e sığar." }
```

---

## 11. Donanım Profilleme ve Optimizasyon Algoritması

**Adım 1 — Kullanılabilir VRAM hesabı:**
`usable_vram = vram_free_mb * 0.85` (güvenlik payı — OS/diğer uygulamalar için %15 rezerv)

**Adım 2 — Katman başına bellek tahmini:**
`layer_size_mb = (model_file_size_mb / total_layers)`

**Adım 3 — GPU'ya sığacak katman sayısı:**
`n_gpu_layers = floor(usable_vram / layer_size_mb)`, KV-cache için context_length'e göre ek pay ayrılır.

**Adım 4 — Kalan katmanların RAM/disk dağılımı:**
`remaining = total_layers - n_gpu_layers`
`ram_layers = min(remaining, floor(usable_ram / layer_size_mb))`
`disk_layers = remaining - ram_layers` (varsa, mmap ile disk streaming)

**Adım 5 — Tahmini hız hesabı (kaba model):**
GPU katman oranı arttıkça token/sn logaritmik olarak artar; disk streaming katmanı varsa ciddi ceza uygulanır (disk I/O darboğazı). Bu tahmin, kullanıcıya "yaklaşık" ibaresiyle sunulur ve ilk gerçek üretimden sonra kalibre edilir.

**Adım 6 — Kullanıcı onayı:** Sistem öneriyi gösterir, kullanıcı "Uygula" veya "Manuel Ayarla" seçer.

---

## 12. Quantization Karar Matrisi

| Quant Seviyesi | Bit/Ağırlık | Kalite Kaybı | Önerilen Kullanım |
|---|---|---|---|
| Q2_K | ~2.6 | Belirgin | Sadece VRAM çok kısıtlıysa (son çare) |
| Q4_K_M | ~4.8 | Az | **Varsayılan öneri** — kalite/boyut dengesi en iyi |
| Q5_K_M | ~5.7 | Çok az | Orta-üst VRAM, kalite öncelikliyse |
| Q6_K | ~6.6 | İhmal edilebilir | Bol VRAM varsa |
| Q8_0 | ~8.5 | Neredeyse yok | Sadece bol kaynaklı sistemler |

Sistem, kullanıcının donanım profiline göre bu tablodan otomatik bir varsayılan seçer ve gerekçesini gösterir ("Q4_K_M seçildi çünkü Q5_K_M ile disk streaming gerekirdi, bu hızı %40 düşürür").

---

## 13. Tasarım Sistemi (Genişletilmiş)

**Renk Paleti (tam):**
- `--bg-base: #14161A`
- `--bg-panel: #1C1F26`
- `--bg-panel-raised: #23262E`
- `--text-primary: #E8E6E1`
- `--text-secondary: #8A8F98`
- `--accent-active: #FFB454` (GPU/işleniyor)
- `--accent-ready: #5EEAD4` (RAM/hazır)
- `--accent-warning: #F87171`
- `--border-hairline: #2A2D35`

**Tipografi Ölçeği:**
- Display: Space Grotesk, 28px/36px, weight 600
- Başlık: Space Grotesk, 18px/24px, weight 500
- Gövde: Inter, 14px/20px, weight 400
- Küçük/etiket: Inter, 12px/16px, weight 500, letter-spacing 0.02em
- Mono (telemetri): JetBrains Mono, 13px/18px, weight 500 (tabular-nums)

**Boşluk Ölçeği:** 4px temel birim → 4/8/12/16/24/32/48/64px

**Hareket İlkeleri:**
- Panel açılış/kapanış: 200ms ease-out
- Telemetri sayaç güncellemeleri: sayı geçişleri yumuşatılır (ease, 150ms), ani sıçrama yok
- Katman dağılım çubuğu: segment değişimi 300ms ease-in-out ile animasyonlu
- Yükleniyor durumları: nabız (pulse) animasyonu, döner spinner değil (enstrüman hissi için)

**İmza Öğesi (detay):** Katman dağılım çubuğu — yatay segmentli çubuk, üç renk (amber=GPU, teal=RAM, gri-mavi=disk), üzerine gelince (hover) her segment kaç katman ve kaç MB olduğunu tooltip ile gösterir.

---

## 14. Bileşen Envanteri

| Bileşen | Sorumluluk | Temel Prop'lar |
|---|---|---|
| `ModelShelf` | Kurulu/indirilebilir model listesi | `models[]`, `onSelect`, `onDownload` |
| `ModelCard` | Tekil model özeti + uyumluluk rozeti | `model`, `hardwareProfile` |
| `ChatConsole` | Aktif sohbet mesaj akışı | `session`, `onSend`, `onStop` |
| `MessageBubble` | Tekil mesaj render | `role`, `content`, `streaming` |
| `TelemetryPanel` | Canlı sistem metrikleri | `snapshot`, `offloadPlan` |
| `LayerDistributionBar` | Katman dağılım görselleştirme | `gpuLayers`, `ramLayers`, `diskLayers` |
| `SettingsModal` | Ayarlar formu | `settings`, `onSave` |
| `HardwareCard` | Donanım profil özeti | `profile` |
| `QuantSelector` | Quantization seçim arayüzü | `availableQuants`, `recommended`, `onSelect` |

---

## 15. Durum Yönetimi Mimarisi

Frontend'de **Zustand** kullanılır (Redux'a göre daha az boilerplate, küçük-orta ölçekli masaüstü uygulaması için yeterli):

```
store/
├── useModelStore.js      // kurulu modeller, indirme durumu
├── useSessionStore.js    // sohbet oturumları, mesajlar
├── useTelemetryStore.js  // WebSocket'ten gelen canlı veri
├── useSettingsStore.js   // kullanıcı ayarları
└── useHardwareStore.js   // donanım profili
```

WebSocket bağlantısı `useTelemetryStore` içinde tek bir singleton olarak yönetilir, tüm bileşenler bu store'dan türetilmiş değerlerle beslenir (gereksiz re-render önlenir, `useShallow` ile).

---

## 16. Hata Yönetimi ve Dayanıklılık

| Hata Sınıfı | Kullanıcıya Gösterim | Sistem Davranışı |
|---|---|---|
| VRAM yetersiz (OOM) | "Bu model şu anki donanımınızda yüklenemedi. Daha düşük quantization deneyin: [Q4_K_M seç]" | Otomatik olarak bir alt quant öner |
| Model dosyası bozuk | "İndirilen dosya bozuk görünüyor. Yeniden indir?" | Checksum doğrulama, otomatik yeniden indirme seçeneği |
| Backend process çöktü | "Motor beklenmedik şekilde durdu, yeniden başlatılıyor…" | Sidecar otomatik restart, son sohbet state'i korunur |
| Ağ bağlantısı yok (indirme sırasında) | "Bağlantı kesildi, indirme %62'de duraklatıldı" | Resumable download ile devam |
| Disk alanı yetersiz | "Bu model için Xgb daha boş alan gerekiyor" | İndirme başlamadan önce ön kontrol |

**Genel ilke:** Hata mesajları asla ham stack trace göstermez; teknik detay "Detayları Göster" ile katlanabilir alanda sunulur (log dosyasına yönlendirme ile).

---

## 17. Güvenlik Tehdit Modeli

| Tehdit | Risk | Önlem |
|---|---|---|
| Yerel API'nin ağ üzerinden yetkisiz erişimi | Orta | Varsayılan `127.0.0.1` bind, ağ açılışı açık onay + opsiyonel API key |
| Kötü amaçlı/manipüle edilmiş GGUF dosyası | Düşük-Orta | Sadece HuggingFace doğrulanmış kaynaklardan indirme, checksum kontrolü |
| Yerel veritabanına yetkisiz erişim (paylaşımlı makine) | Düşük | Sohbet geçmişi opsiyonel şifreleme (SQLCipher) |
| Sidecar process'in ayrıcalık yükseltmesi | Düşük | Backend en düşük OS ayrıcalığıyla çalışır, dosya erişimi sadece uygulama klasörleriyle sınırlı |
| Tedarik zinciri (bağımlılık) riski | Orta | Bağımlılıklar kilitlenir (lockfile), düzenli güvenlik taraması (Dependabot/pip-audit) |

---

## 18. Test Stratejisi

- **Birim testleri:** `memory_manager.py` (offload hesaplama algoritması) ve `hardware_profile.py` için %90+ kapsam hedefi — bunlar ürünün "beyni".
- **Entegrasyon testleri:** FastAPI endpoint'leri, mock model dosyalarıyla (gerçek 70B indirmeden CI'da test).
- **Uçtan uca (E2E):** Playwright ile Tauri WebView üzerinde temel akışlar (model yükle → sohbet et → telemetri güncelleniyor mu).
- **Donanım matrisi testleri:** Manuel/haftalık — düşük VRAM (4GB), orta (8GB), yüksek (24GB) simülasyon profilleriyle offload planlayıcı doğrulama.
- **Performans regresyon testleri:** Her sürümde referans modelle token/sn ölçümü, önceki sürümle karşılaştırma.

---

## 19. Gözlemlenebilirlik ve Loglama

- Yerel log dosyası: `~/.ai-runner/logs/backend.log` (rotating, max 10MB x 5 dosya)
- Log seviyeleri: DEBUG (İleri Mod açıkken), INFO (varsayılan), ERROR
- Her model yükleme olayı yapılandırılmış (structured JSON) log satırı üretir — sorun giderme için dışa aktarılabilir "Tanı Raporu" (Diagnostic Report) özelliği: donanım profili + son 50 log satırı + aktif offload planı tek dosyada.

---

## 20. Paketleme, Dağıtım, Güncelleme

- Backend: PyInstaller ile tek binary'e derlenir, Tauri `resources` içine gömülür.
- Frontend + Tauri: `tauri build` ile platforma özel installer (`.msi`, `.dmg`, `.AppImage`/`.deb`).
- Kod imzalama: Windows (Authenticode) ve macOS (Apple Developer ID) için imzalama pipeline'ı.
- Otomatik güncelleme: Tauri'nin yerleşik updater'ı, GitHub Releases üzerinden delta güncelleme.
- CI/CD: GitHub Actions — her PR'da lint + birim test, `main`'e merge'de otomatik nightly build, tag push'ta resmi release.

---

## 21. Dizin Yapısı (Genişletilmiş)

```
ai-runner/
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── hardware_profile.py
│   │   ├── model_loader.py
│   │   ├── memory_manager.py
│   │   ├── inference_engine.py
│   │   └── quant_matrix.py
│   ├── models/
│   │   └── model_manager.py
│   ├── api/
│   │   ├── routes_models.py
│   │   ├── routes_chat.py
│   │   ├── routes_settings.py
│   │   └── ws_telemetry.py
│   ├── db/
│   │   ├── schema.sql
│   │   └── session_store.py
│   ├── tests/
│   │   ├── test_memory_manager.py
│   │   └── test_hardware_profile.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/ (Bölüm 14'teki envanter)
│   │   ├── store/ (Bölüm 15)
│   │   ├── styles/tokens.css
│   │   ├── i18n/ (tr.json, en.json)
│   │   └── main.jsx
│   ├── e2e/
│   └── package.json
├── src-tauri/
│   ├── tauri.conf.json
│   ├── Cargo.toml
│   └── src/main.rs
├── docs/
│   ├── PROJECT_SPEC_V2.md
│   └── diagnostic-report-format.md
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
└── README.md
```

---

## 22. Yol Haritası (Süre Tahminli)

| Faz | İçerik | Tahmini Süre |
|---|---|---|
| Faz 1 | Çekirdek motor: donanım profilleme + model yükleme (CLI doğrulama) | 1–2 hafta |
| Faz 2 | Model yönetimi: HF indirme, yerel kütüphane, quant matrisi | 1 hafta |
| Faz 3 | Arayüz iskeleti: React + Tauri sarmalama, statik paneller | 1–2 hafta |
| Faz 4 | Canlı telemetri: WebSocket streaming, katman dağılım çubuğu | 1 hafta |
| Faz 5 | Cilalama: hata durumları, ayarlar, i18n, erişilebilirlik | 1–2 hafta |
| Faz 6 | Paketleme/dağıtım: installer, imzalama, auto-update, CI/CD | 1 hafta |
| **Faz 7a** | Hızlı kazanımlar: Prompt Kütüphanesi (FR-908), Kişilik Preset'leri (FR-918), Model Güncelleme Takibi (FR-917), Gizlilik Paneli (FR-920), Isınma Modu (FR-904) | 1–2 hafta |
| **Faz 7b** | RAG ve araçlar: Yerel RAG (FR-909), Araç Çağırma (FR-910), Kod Sandbox (FR-914) | 3–4 hafta |
| **Faz 7c** | Çoklu model mimarisi: Model Router (FR-903), Model Arenası (FR-912), Katman Önbellekleme (FR-902), Adaptif Quantization (FR-901) | 3–4 hafta |
| **Faz 7d** | Ses ve dış entegrasyonlar: Sesli Arayüz (FR-911), VSCode Eklentisi (FR-915), LAN Paylaşımı (FR-916), Enerji Profilleri (FR-919) | 2–3 hafta |
| **Faz 8** | Topluluk ve ekosistem: Performans Veritabanı (FR-905), Liderlik Tablosu (FR-906), Eklenti Pazarı (FR-907), Yerel LoRA İnce Ayar (FR-913) | Belirlenecek — her biri ayrı fizibilite gerektirir |

> Detaylı özellik tanımları için bkz. **Bölüm 26 — Genişletilmiş Özellik Kataloğu** ve **Bölüm 27 — Öncelik/Karmaşıklık Matrisi**.

---

## 23. Başarı Metrikleri

- **Kurulumdan ilk başarılı üretime geçen süre:** <5 dakika (model indirme hariç)
- **Donanım uyumsuzluğu kaynaklı çökme oranı:** <%1 (offload planlayıcı öncesi tahmin doğruluğu ile ölçülür)
- **Kullanıcı tarafından bildirilen "beklenmedik yavaşlık" oranı:** telemetri sayesinde azaltılması hedeflenir (kullanıcı neden yavaş olduğunu görebildiği için destek talebi azalır)

---

## 24. Risk Kayıt Defteri

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| llama.cpp API değişiklikleri bağımlılığı kırar | Orta | Yüksek | Sürüm kilitleme, düzenli uyumluluk testi |
| Çeşitli GPU sürücü versiyonlarında tutarsız davranış | Orta | Orta | Geniş donanım test matrisi, kullanıcı geri bildirim kanalı |
| Disk streaming performansı kullanıcı beklentisini karşılamaz | Yüksek | Orta | Beklenti yönetimi: önizleme ekranında net hız tahmini ve uyarı |
| Tauri ekosistemi olgunluk riski (Electron'a göre daha genç) | Düşük | Orta | Electron'a geçiş için mimari soyutlama (backend zaten ayrı process) |

---

## 25. Açık Sorular

1. Çoklu model eşzamanlı yükleme (v1.0'da tek model sınırı) ne zaman genişletilecek?
2. Şifreleme (SQLCipher) varsayılan mı yoksa opsiyonel mi olmalı?
3. Windows üzerinde AMD GPU (ROCm) desteği ilk sürümde mi yoksa sonraki fazda mı?

---

## 26. Genişletilmiş Özellik Kataloğu (Faz 7+)

Bu bölüm, v1.0 sonrası ürünü farklılaştıracak 20 özelliği; teknik yaklaşım, mimari etki ve bağımlılıklarıyla birlikte tanımlar. Her madde bir gereksinim ID'si taşır ve Bölüm 22'deki yol haritasına bağlanır.

### 26.1 Akıllı Optimizasyon

**FR-901 — Adaptif Quantization**
- *Açıklama:* Bellek baskısı arttıkça (uzayan konuşma, context dolması) sistem otomatik olarak daha düşük quant seviyesine geçer, kesinti hissettirmeden.
- *Teknik Yaklaşım:* MemoryManager VRAM/RAM kullanımını sürekli izler; eşik (örn. %90) aşıldığında düşük quant'lı model versiyonu arka planda paralel yüklenir, hazır olduğunda "hot-swap" ile devralır; KV-cache state'i yeniden oluşturulur.
- *Mimari Etkisi:* InferenceEngine'de geçici çift-model yönetimi; MemoryManager'a eşik tabanlı tetikleyici eklenir.
- *Bağımlılık:* Modelin birden fazla quant versiyonunun yerelde bulunması ya da anlık indirilebilir olması.
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta

**FR-902 — Katman Önbellekleme (LRU Layer Cache)**
- *Açıklama:* Disk streaming yapılan katmanları NVMe üzerinde akıllı LRU önbellekte tutarak tekrarlayan erişimlerde gecikmeyi azaltma.
- *Teknik Yaklaşım:* Memory-mapped dosya erişimi + LRU eviction politikası; sık erişilen katmanlar için ikinci seviye önbellek katmanı.
- *Mimari Etkisi:* MemoryManager'a önbellek alt modülü eklenir; disk I/O metrikleri Telemetri Paneli'ne (Bölüm 6.4) yansır.
- *Karmaşıklık:* Orta-Yüksek · *Öncelik:* Orta

**FR-903 — Model Router / Orkestrasyon**
- *Açıklama:* Basit sorular küçük/hızlı modele, karmaşık sorular büyük modele otomatik yönlendirilir.
- *Teknik Yaklaşım:* Hafif bir sınıflandırıcı (heuristic kurallar veya küçük yardımcı model) prompt karmaşıklığını skorlar; kullanıcı yönlendirme kararını her zaman override edebilir.
- *Mimari Etkisi:* Yeni `RouterEngine` modülü; **FR-307'deki "tek aktif model" kısıtının Faz 7'de gözden geçirilmesini gerektirir** — çoklu model bellek bütçelemesi MemoryManager'a eklenmelidir.
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta

**FR-904 — Isınma (Warm-up) Modu**
- *Açıklama:* Sık kullanılan model, uygulama açılışında arka planda önceden kısmi yüklenerek ilk yanıt gecikmesi (TTFT) sıfıra yakın hale getirilir.
- *Teknik Yaklaşım:* Kullanım geçmişine göre en sık kullanılan model belirlenir, boş promptla KV-cache önceden ısıtılır.
- *Mimari Etkisi:* Başlangıç sürecine opsiyonel arka plan görevi; ayarlardan açılıp kapatılabilir (ek VRAM/RAM maliyeti kullanıcıya açıkça gösterilir).
- *Karmaşıklık:* Düşük-Orta · *Öncelik:* Orta

### 26.2 Topluluk ve Veri

**FR-905 — Anonim Performans Veritabanı**
- *Açıklama:* Kullanıcılar gönüllü olarak donanım+model+hız verisini paylaşır; sistem "senin donanımına benzer kullanıcılar bu modeli şu hızda çalıştırıyor" önerisi sunar.
- *Teknik Yaklaşım:* Opt-in telemetri; ayrı bir toplama servisi (`community-stats-api`) gerektirir. Veri, donanım modeli + performans metriklerinden oluşur, kullanıcı kimliği taşımaz.
- *Mimari Etkisi:* Yerel-öncelikli mimariye ilk kez bir **bulut bileşeni** eklenir — Bölüm 10 (Güvenlik ve Gizlilik) ilkeleriyle uyumlu, açık ve ayrıştırılabilir rıza akışı zorunludur.
- *Karmaşıklık:* Yüksek (yeni sunucu altyapısı) · *Öncelik:* Düşük

**FR-906 — Topluluk Liderlik Tablosu**
- *Açıklama:* Donanım/model bazlı performans kıyaslama panosu.
- *Teknik Yaklaşım:* FR-905 verisini tüketen salt-okunur görünüm (uygulama içi sekme veya bağımsız web sayfası).
- *Mimari Etkisi:* Doğrudan FR-905'e bağımlı, ondan önce yapılamaz.
- *Karmaşıklık:* Orta · *Öncelik:* Düşük

**FR-907 — Eklenti Pazarı (Marketplace)**
- *Açıklama:* Üçüncü parti eklentilerin keşfedilebileceği, kurulabileceği galeri arayüzü.
- *Teknik Yaklaşım:* FR-801'deki eklenti manifestosunu temel alan imzalı paket dağıtım sistemi.
- *Mimari Etkisi:* Eklenti sandbox'lama zorunlu hale gelir — Bölüm 17 tehdit modeline "kötü amaçlı/güvenilmez eklenti" satırı eklenmelidir.
- *Karmaşıklık:* Yüksek · *Öncelik:* Düşük

**FR-908 — Prompt Kütüphanesi**
- *Açıklama:* Sık kullanılan sistem promptlarını kaydetme, kategorize etme, dışa/içe aktarma ve paylaşma.
- *Teknik Yaklaşım:* SQLite'a yeni `prompt_templates` tablosu (başlık, içerik, kategori, etiketler); JSON dışa/içe aktarma.
- *Mimari Etkisi:* Düşük — mevcut veritabanı şemasına ek tablo, yeni endpoint seti (`/api/prompts`).
- *Karmaşıklık:* Düşük · *Öncelik:* **Yüksek** (düşük efor, yüksek algılanan değer)

### 26.3 Fonksiyonel Genişleme

**FR-909 — Yerel RAG Entegrasyonu**
- *Açıklama:* PDF/doküman yükleyip yerel embedding modeliyle sohbete bağlama (belge üzerinden soru-cevap).
- *Teknik Yaklaşım:* Hafif embedding modeli (örn. ~130MB sınıfı bir model) + gömülü vektör deposu (SQLite tabanlı vektör uzantısı); doküman parçalama (chunking) pipeline'ı.
- *Mimari Etkisi:* Yeni `RAGManager` modülü; `ChatSession` şemasına (Bölüm 9) `attached_documents[]` alanı eklenir.
- *Karmaşıklık:* Yüksek · *Öncelik:* **Yüksek** (belirgin rekabet farkı)

**FR-910 — Araç Çağırma (Tool Use)**
- *Açıklama:* Modelin hesap makinesi, dosya okuma, yerel arama gibi araçları çağırabilmesi (function calling).
- *Teknik Yaklaşım:* Function-calling destekleyen modellerde JSON şema tabanlı araç tanımlama; InferenceEngine'e ReAct-benzeri araç çalıştırma döngüsü eklenir.
- *Mimari Etkisi:* Güvenlik kritik — araçlar izole ortamda çalışmalı (FR-914 ile ortak sandbox altyapısı paylaşılabilir).
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta-Yüksek

**FR-911 — Sesli Arayüz**
- *Açıklama:* Yerel STT/TTS motorlarıyla tamamen offline sesli sohbet.
- *Teknik Yaklaşım:* Hafif yerel konuşma-tanıma ve seslendirme motorları; ek VRAM/CPU bütçesi Donanım Profili hesaplamasına (Bölüm 11) dahil edilmelidir.
- *Mimari Etkisi:* Yeni `AudioEngine` modülü; ana LLM ile kaynak paylaşımı planlaması MemoryManager'a eklenir.
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta

**FR-912 — Model Arenası**
- *Açıklama:* Aynı prompt'u 2-3 modelde paralel çalıştırıp kalite/hız kıyaslaması yapma.
- *Teknik Yaklaşım:* InferenceEngine'in çoklu model instance yönetebilmesi gerekir (FR-903 ile ortak mimari genişleme); yan yana karşılaştırma bileşeni (`ArenaView`).
- *Mimari Etkisi:* Bellek bütçelemesi kritik hale gelir — birden fazla model VRAM/RAM'i paylaşır.
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta

**FR-913 — Yerel LoRA/QLoRA İnce Ayar**
- *Açıklama:* Küçük veri setleriyle düşük VRAM'de kişiselleştirilmiş model eğitimi.
- *Teknik Yaklaşım:* PEFT/QLoRA kütüphaneleri; eğitim arka planda ayrı bir "training job" olarak yürütülür, ilerleme takip ekranı.
- *Mimari Etkisi:* Projenin "yalnızca inference" kapsamını aşar — ayrı bir `TrainingEngine` modülü ve önemli yeni bağımlılıklar gerektirir. **Ayrı bir alt-proje olarak değerlendirilmesi önerilir.**
- *Karmaşıklık:* Çok Yüksek · *Öncelik:* Düşük

**FR-914 — Kod Çalıştırma Sandbox'ı**
- *Açıklama:* Sohbette üretilen kodu izole bir ortamda çalıştırabilme.
- *Teknik Yaklaşım:* Konteyner tabanlı izolasyon, kaynak/süre limitleri, varsayılan olarak ağ erişimi kapalı.
- *Mimari Etkisi:* Bölüm 17 tehdit modeline "sandbox kaçışı" riski eklenir; güvenlik incelemesi zorunlu.
- *Karmaşıklık:* Yüksek · *Öncelik:* Orta

### 26.4 Entegrasyon

**FR-915 — VSCode Eklentisi**
- *Açıklama:* Yerel API'ye bağlanan, kod tamamlama sağlayan editör eklentisi.
- *Teknik Yaklaşım:* Mevcut FR-501 (OpenAI-uyumlu API) üzerinden çalışan bağımsız bir TypeScript eklenti projesi.
- *Mimari Etkisi:* Ana uygulamadan bağımsız, ayrı repo/paket olarak geliştirilir; ortak nokta yalnızca API sözleşmesidir.
- *Karmaşıklık:* Orta · *Öncelik:* **Yüksek** (geliştirici kullanıcı kitlesi için doğrudan değer)

**FR-916 — LAN Üzerinden Model Paylaşımı**
- *Açıklama:* Küçük ekiplerin modeli tekrar tekrar indirmeden yerel ağdan senkronize edebilmesi.
- *Teknik Yaklaşım:* Yerel ağda otomatik keşif (mDNS/Bonjour) + parçalı dosya paylaşım protokolü.
- *Mimari Etkisi:* FR-503'teki "yalnızca localhost" varsayılanına açık kullanıcı onayı gerektiren bir istisna getirir.
- *Karmaşıklık:* Orta-Yüksek · *Öncelik:* Düşük-Orta

**FR-917 — Otomatik Model Güncelleme Takibi**
- *Açıklama:* Kurulu modelin daha iyi/güncel bir quant versiyonu yayınlandığında kullanıcıyı bilgilendirme.
- *Teknik Yaklaşım:* Periyodik HuggingFace API kontrolü (sıklık ayarlanabilir), yerel model meta verisiyle karşılaştırma.
- *Mimari Etkisi:* ModelManager'a zamanlanmış görev (scheduler) eklenir.
- *Karmaşıklık:* Düşük · *Öncelik:* Orta

### 26.5 Kullanıcı Deneyimi

**FR-918 — Model "Kişilik" Preset'leri**
- *Açıklama:* Yaratıcı yazar, kodlayıcı, analist gibi hazır parametre + sistem promptu kombinasyonları.
- *Teknik Yaklaşım:* FR-908 (Prompt Kütüphanesi) altyapısı üzerine kurulu, önceden tanımlı preset koleksiyonu + kullanıcı özel preset oluşturma.
- *Mimari Etkisi:* Düşük — mevcut ayarlar/prompt altyapısını doğrudan kullanır.
- *Karmaşıklık:* Düşük · *Öncelik:* **Yüksek**

**FR-919 — Enerji Profilleri**
- *Açıklama:* Laptop kullanıcıları için "sessiz mod" (güç/fan limitli) ve "performans modu" seçimi.
- *Teknik Yaklaşım:* Thread sayısı, batch boyutu ve (destekleniyorsa) GPU güç limitini profil olarak gruplama.
- *Mimari Etkisi:* HardwareProfiler'a pil/şarj durumu tespiti eklenir; pil modundayken otomatik sessiz mod önerisi.
- *Karmaşıklık:* Orta · *Öncelik:* Orta

**FR-920 — Gizlilik Denetim Paneli**
- *Açıklama:* Hangi API isteğinin ne zaman/nereden geldiğini gösteren ağ trafiği kayıt ekranı.
- *Teknik Yaklaşım:* API middleware katmanında her isteği (kaynak IP, zaman damgası, endpoint) loglama; Ayarlar içinde "Gizlilik" sekmesinde görselleştirme.
- *Mimari Etkisi:* Bölüm 19 (Gözlemlenebilirlik) ile doğrudan entegre; düşük karmaşıklık.
- *Karmaşıklık:* Düşük · *Öncelik:* Orta (kullanıcı güveni için değerli)

---

## 27. Öncelik / Karmaşıklık Matrisi (Faz 7+)

Aşağıdaki matris, 20 özelliği **etki** (kullanıcı değeri) ve **efor** (geliştirme karmaşıklığı) eksenlerinde dört gruba ayırır. Yol haritası önceliklendirmesi bu matrise dayanır.

| Grup | Tanım | Özellikler |
|---|---|---|
| 🟢 **Hızlı Kazanımlar** (Düşük efor, Yüksek etki) | Önce bunlar yapılmalı | FR-908 (Prompt Kütüphanesi), FR-918 (Kişilik Preset'leri), FR-917 (Güncelleme Takibi), FR-920 (Gizlilik Paneli) |
| 🔵 **Stratejik Yatırımlar** (Yüksek/Orta efor, Yüksek etki) | Farklılaşma yaratır, planlı yatırım gerekir | FR-909 (Yerel RAG), FR-910 (Araç Çağırma), FR-915 (VSCode Eklentisi), FR-911 (Sesli Arayüz) |
| 🟡 **Dolgu İşler** (Düşük-Orta efor, Orta etki) | Boş zamanlarda değerlendirilebilir | FR-904 (Isınma Modu), FR-902 (Katman Önbellekleme), FR-919 (Enerji Profilleri) |
| 🔴 **Dikkatli Değerlendir** (Yüksek efor, Orta/Düşük etki) | ROI belirsiz, ayrı fizibilite gerekir | FR-903 (Model Router), FR-912 (Model Arenası), FR-905/906 (Topluluk Verisi/Tablosu), FR-907 (Eklenti Pazarı), FR-913 (LoRA İnce Ayar), FR-914 (Kod Sandbox), FR-916 (LAN Paylaşımı) |

**Önerilen uygulama sırası:** 🟢 Hızlı Kazanımlar (Faz 7a) → 🔵 Stratejik Yatırımlar'ın en yüksek etkili ikisi (Faz 7b: RAG + Araç Çağırma) → 🟡 Dolgu İşler mimari genişlemeyle birlikte (Faz 7c) → 🔴 grubundan yalnızca gerçek kullanıcı talebi doğrulanan maddeler (Faz 8, vaka bazlı değerlendirme).

---

## 28. Sözlük

- **GGUF:** llama.cpp ekosisteminin kullandığı, quantize edilmiş model ağırlıklarını tek dosyada barındıran format.
- **Offload:** Modelin bir kısmının GPU yerine RAM veya diskte tutulup işlenmesi.
- **TTFT:** Time To First Token — isteğin gönderilmesinden ilk token'ın üretilmesine kadar geçen süre.
- **Speculative Decoding:** Küçük bir modelin taslak ürettiği, büyük modelin bu taslağı doğruladığı hızlandırma tekniği.
- **Sidecar Process:** Ana uygulamayla birlikte paketlenen, ayrı çalışan yardımcı süreç (burada Python backend).

---

**Bu doküman, geliştirme sürecinde yaşayan bir referans olarak kullanılmalı; her faz tamamlandığında ilgili gereksinim ID'leri "tamamlandı" olarak işaretlenmelidir.**
