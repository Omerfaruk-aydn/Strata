@echo off
echo 🚀 AI Runner - Sifir VRAM Modu Baslatiliyor...
echo -----------------------------------------------
echo [INFO] Bu modda Tauri arayuzu tamamen CPU ile cizilir.
echo [INFO] WebView2 VRAM kullanimi %%100 engellenir (~400MB tasarruf).
echo -----------------------------------------------

:: WebView2 GPU engelleme degiskenleri
set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--disable-gpu --disable-gpu-compositing
set TAURI_DISABLE_GPU=1
set TAURI_FORCE_CPU=1

:: Uygulamayi baslat
npm run dev
