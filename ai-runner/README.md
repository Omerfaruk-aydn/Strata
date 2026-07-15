# AI Runner

AI Runner, GGUF biçimindeki büyük dil modellerini yerel donanımda çalıştırmak için geliştirilmiş React, FastAPI, llama.cpp ve Tauri tabanlı bir masaüstü uygulamasıdır. Model çıkarımı çevrimdışı çalışır; yalnızca Hugging Face araması ve model indirme işlemleri ağ erişimi gerektirir.

## Öne çıkan özellikler

- Hugging Face üzerinde gerçek GGUF araması; tek dosyalı quant seçimi, devam ettirilebilir HTTP Range indirme, gerçek hız/ETA ve iptal desteği
- İndirme öncesi disk/önbellek sınırı kontrolü, GGUF magic doğrulaması ve SHA-256 kayıtları
- GPU/RAM katman planlama, ana GPU seçimi ve çoklu GPU `tensor_split` desteği
- KV cache quantization, Flash Attention, context shifting ve prompt lookup decoding
- Kesilebilir token akışı, sohbet kalıcılığı ve Markdown/JSON dışa aktarma
- OpenAI uyumlu `/v1/chat/completions` ve `/v1/models` uçları; gerçek token kullanımı ve bitiş nedeni
- WebSocket tabanlı GPU, VRAM, RAM ve üretim telemetrisi
- İsteğe bağlı Bearer/API-key koruması, tarayıcı origin kontrolü ve güvenli loopback varsayılanı
- Python backend'i ve CUDA çalışma zamanı DLL'lerini içeren Tauri 2 Windows paketi

## Proje yapısı

```text
ai-runner/
├── backend/
│   ├── api/                 FastAPI rotaları, kimlik doğrulama ve WebSocket
│   ├── core/                çıkarım, donanım, bellek ve sistem optimizasyonu
│   ├── db/                  SQLite şeması ve asenkron veri erişimi
│   ├── models/              Hugging Face indirme ve yerel model kütüphanesi
│   └── tests/               birim ve API bütünleşme testleri
├── scripts/                 sidecar, duman testi ve masaüstü derleme betikleri
├── src/                     React 18 + Zustand kullanıcı arayüzü
├── src-tauri/               Tauri/Rust süreç ve paketleme katmanı
├── backend_sidecar.py       dondurulmuş backend giriş noktası
└── package.json             frontend ve masaüstü komutları
```

## Gereksinimler

- Python 3.11 veya 3.12
- Node.js 20 veya üzeri
- Model boyutuna uygun RAM ve disk alanı; 16 GB RAM önerilir
- İsteğe bağlı NVIDIA GPU ve hedef CUDA sürümüyle uyumlu `llama-cpp-python` wheel'i
- Masaüstü paketi için Rust stable, MSVC C++ Build Tools ve WebView2

Python çalışma zamanı sürümleri [backend/requirements.txt](backend/requirements.txt) içinde sabitlenmiştir. CUDA, Metal veya CPU hedefinize uygun `llama-cpp-python==0.3.34` paketini seçili Python ortamına kurun; masaüstü betiği bu seçimi sidecar ortamına taşır.

## Geliştirme kurulumu

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements-dev.txt

npm ci
```

Backend ve frontend'i iki ayrı terminalde başlatın:

```powershell
python -m backend.main
```

```powershell
npm run dev
```

Varsayılan adresler:

- Arayüz: `http://localhost:5173`
- API: `http://127.0.0.1:8420`
- OpenAPI: `http://127.0.0.1:8420/docs`

Backend farklı bir portta başlatılabilir:

```powershell
python -m backend.main --host 127.0.0.1 --port 9000
```

## Masaüstü paketi

```powershell
# PyInstaller sidecar'ını oluştur
npm run sidecar:build

# Dondurulmuş .exe'yi boş bir portta başlatıp HTTP sağlık kontrolü yap
npm run sidecar:smoke

# Frontend + sidecar + Tauri release paketleri
npm run desktop:build
```

`sidecar:build`, çağrıldığı Python ortamındaki seçilmiş llama.cpp wheel'ini korur, gerekli runtime paketlerini doğrular, NVIDIA CUDA DLL'lerini pakete ekler ve çözülemeyen native kütüphane varsa derlemeyi durdurur. Üretilen kurucular `src-tauri/target/release/bundle/` altında bulunur.

## Test ve kalite kapıları

```powershell
# 175 test ve en az %70 kaynak-kodu kapsamı
python -m pytest backend\tests -q --cov=backend --cov-config=.coveragerc --cov-report=term-missing

# Frontend üretim derlemesi ve bağımlılık denetimi
npm audit --audit-level=moderate
npm run build

# Rust/Tauri derleme denetimi
cargo check --manifest-path src-tauri\Cargo.toml --locked
```

Doğrulanan mevcut sonuç: **175 test geçti, backend kaynak kapsamı %74,5, npm ve doğrudan Python bağımlılıklarında bilinen güvenlik açığı 0**. Aynı kontroller [.github/workflows/ci.yml](../.github/workflows/ci.yml) üzerinden Windows CI'da çalışır.

## API güvenliği

API varsayılan olarak yalnızca `127.0.0.1` üzerinde dinler. Ayarlardan bir API anahtarı tanımlandığında tüm `/api/*`, `/v1/*` ve telemetri WebSocket erişimleri korunur.

```bash
curl http://127.0.0.1:8420/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Loopback dışındaki bir adrese bağlanmak için hem "Yerel Ağ Erişimine İzin Ver" seçeneği hem de boş olmayan bir API anahtarı zorunludur. Anahtar ortam değişkeniyle de verilebilir:

```powershell
$env:AI_RUNNER_API_KEY = "uzun-rastgele-bir-anahtar"
python -m backend.main --host 0.0.0.0 --allow-network
```

Host veya port değişikliği çalışan bağlantıyı kesmez; bir sonraki uygulama açılışında uygulanır.

## OpenAI uyumlu örnek

Önce arayüzden yerel bir modeli yükleyin:

```bash
curl http://127.0.0.1:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Merhaba!"}],
    "max_tokens": 256,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

API anahtarı yapılandırılmadıysa `Authorization` başlığı gerekli değildir.

## Veri konumları

- SQLite ve loglar: `%USERPROFILE%\.ai-runner\`
- Varsayılan modeller: `%USERPROFILE%\.ai-runner\models\`
- İndirme sırasında kısmi dosya: `*.gguf.part`
- Frontend API bağlantı ayarı: WebView/localStorage

## Bilinen sınırlar

- Çok parçalı/sharded GGUF depoları henüz indirilmez; tek dosyalı bir quant seçilmelidir.
- Spekülatif mod ikinci bir draft modeli yüklemez; llama.cpp'nin desteklediği prompt lookup decoding kullanılır.
- Model dosyaları ve oluşturulan sidecar ikilileri Git'e alınmaz.

## Klavye kısayolları

| Kısayol | İşlev |
|---|---|
| `Ctrl+N` | Yeni sohbet |
| `Ctrl+K` | Model aramasına odaklan |
| `Ctrl+,` | Ayarları aç |
| `Ctrl+Shift+O` | Sistem optimizasyonunu aç |
| `Ctrl+B` | Sol paneli aç/kapat |
| `Ctrl+.` | Telemetri panelini aç/kapat |
| `Enter` | Mesaj gönder |
| `Shift+Enter` | Yeni satır |
| `Esc` | Üretimi durdur |

## Lisans

MIT
