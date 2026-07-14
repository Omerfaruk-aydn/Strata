# AI Runner — Yerel LLM Çalıştırma Platformu

<div align="center">
  <h3>🤖 Kısıtlı Donanımda Büyük Dil Modellerini Çalıştırın</h3>
  <p>GPU/RAM/Disk akıllı katman dağıtımı · OpenAI uyumlu API · Tam offline</p>
</div>

---

## 🚀 Özellikler

| Özellik | Durum |
|---------|-------|
| 🔍 HuggingFace Hub model arama | ✅ |
| 📥 Devam ettirilebilir model indirme | ✅ |
| 🧠 GPU/RAM/Disk akıllı offload planlaması | ✅ |
| 🟢🟡🔴 Uyumluluk rozetleri | ✅ |
| 💬 Gerçek zamanlı token akışı (SSE) | ✅ |
| 📊 WebSocket telemetri paneli | ✅ |
| 🌙 Karanlık/Aydınlık tema | ✅ |
| 🇹🇷🇬🇧 Türkçe/İngilizce arayüz | ✅ |
| 💾 Sohbet geçmişi (SQLite) | ✅ |
| 📤 Markdown/JSON dışa aktarma | ✅ |
| ⚙️ OpenAI uyumlu API | ✅ |
| ⌨️ Klavye kısayolları | ✅ |

## 🏗️ Mimari

```
ai-runner/
├── backend/                    # Python FastAPI sunucusu
│   ├── core/
│   │   ├── hardware_profile.py   # GPU/CPU/RAM/Disk tespiti
│   │   ├── memory_manager.py     # 6-adım offload planlama algoritması
│   │   ├── quant_matrix.py       # Quantization karar matrisi
│   │   ├── model_loader.py       # GGUF dosya doğrulama
│   │   └── inference_engine.py   # llama-cpp-python wrapper
│   ├── models/
│   │   └── model_manager.py      # HuggingFace Hub entegrasyonu
│   ├── api/
│   │   ├── routes_models.py      # Model yönetimi API
│   │   ├── routes_chat.py        # OpenAI uyumlu chat API
│   │   ├── routes_settings.py    # Ayarlar API
│   │   └── ws_telemetry.py       # WebSocket telemetri
│   ├── db/
│   │   ├── schema.sql            # SQLite şeması
│   │   └── session_store.py      # Async DB katmanı
│   ├── tests/                    # 96 birim test
│   └── main.py                   # FastAPI uygulama
│
├── src/                        # React 18 + Vite arayüzü
│   ├── components/
│   │   ├── ModelShelf.jsx        # Sol panel: model rafı
│   │   ├── ModelCard.jsx         # Model kartı + rozet
│   │   ├── ChatConsole.jsx       # Merkez: sohbet konsolu
│   │   ├── MessageBubble.jsx     # Markdown + kod vurgulama
│   │   ├── SessionList.jsx       # Sohbet geçmiş listesi
│   │   ├── TelemetryPanel.jsx    # Sağ panel: telemetri
│   │   ├── LayerDistributionBar  # Katman dağılım barı
│   │   ├── HardwareCard.jsx      # Donanım bilgisi
│   │   └── SettingsModal.jsx     # Ayarlar modali
│   ├── store/
│   │   ├── useModelStore.js      # Model state yönetimi
│   │   ├── useSessionStore.js    # Sohbet state yönetimi
│   │   ├── useTelemetryStore.js  # WebSocket telemetri
│   │   ├── useSettingsStore.js   # Kullanıcı ayarları
│   │   └── useHardwareStore.js   # Donanım profili
│   ├── i18n/
│   │   ├── tr.json               # Türkçe çeviriler
│   │   ├── en.json               # İngilizce çeviriler
│   │   └── useTranslation.js     # i18n hook
│   └── styles/
│       ├── tokens.css            # Design token sistemi
│       └── global.css            # Global stiller
│
└── src-tauri/                  # Tauri 2.0 kabuk (Rust)
    ├── src/main.rs               # Pencere + sidecar yönetimi
    └── tauri.conf.json           # Tauri konfigürasyonu
```

## ⚡ Hızlı Başlangıç

### 1. Backend'i Başlatın

```bash
# Sanal ortam oluşturun (önerilir)
python -m venv .venv
.venv\Scripts\activate

# Bağımlılıkları kurun
pip install -r backend/requirements.txt

# Sunucuyu başlatın
python -m backend.main
# → http://127.0.0.1:8420
```

### 2. Frontend'i Başlatın

```bash
npm install
npm run dev
# → http://localhost:5173
```

### 3. (İsteğe Bağlı) Tauri Masaüstü Uygulaması

```bash
# Rust ve Tauri CLI kurulu olmalı
npm run tauri dev
```

## 🧪 Testler

```bash
# Tüm testleri çalıştır
python -m pytest backend/tests/ -v --asyncio-mode=auto

# Coverage raporu
python -m pytest backend/tests/ --cov=backend --cov-report=term-missing

# Sadece belirli testler
python -m pytest backend/tests/test_memory_manager.py -v
```

**Test Sonuçları:** 96/96 ✅

| Modül | Testler | Durum |
|-------|---------|-------|
| Hardware Profile | 13 | ✅ |
| Memory Manager | 39 | ✅ |
| Inference Engine | 19 | ✅ |
| Session Store | 25 | ✅ |

## ⌨️ Klavye Kısayolları

| Kısayol | İşlev |
|---------|-------|
| `Ctrl+N` | Yeni sohbet |
| `Ctrl+K` | Model arama |
| `Ctrl+,` | Ayarlar |
| `Ctrl+B` | Sol panel aç/kapat |
| `Ctrl+.` | Telemetri paneli aç/kapat |
| `Enter` | Mesaj gönder |
| `Shift+Enter` | Yeni satır |
| `Esc` | Üretimi durdur |

## 🔌 API

OpenAI uyumlu endpoint:

```bash
# Sohbet tamamlama
curl http://127.0.0.1:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Llama-3-8B-GGUF",
    "messages": [{"role": "user", "content": "Merhaba!"}],
    "stream": true
  }'

# Model listesi
curl http://127.0.0.1:8420/v1/models

# Donanım profili
curl http://127.0.0.1:8420/api/hardware/profile

# Model arama
curl http://127.0.0.1:8420/api/models/search?q=llama
```

## 🗺️ Yol Haritası

- [ ] **Faz 6:** Tauri sidecar ile tam masaüstü paketleme
- [ ] **Faz 7:** Çoklu GPU desteği (Round-robin dengeleme)
- [ ] **Faz 8:** Prompt şablonları ve RAG entegrasyonu
- [ ] **Faz 9:** Plugin sistemi

## 📋 Teknik Gereksinimler

- **Python:** 3.11+
- **Node.js:** 18+
- **RAM:** Min 8 GB (16 GB önerilir)
- **Disk:** Model başına 2–70 GB
- **GPU:** İsteğe bağlı (NVIDIA CUDA destekli önerilir)

---

<div align="center">
  <sub>AI Runner Kurumsal Spesifikasyon v3.0'a göre oluşturulmuştur</sub>
</div>
