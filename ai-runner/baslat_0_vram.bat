@echo off
echo AI Runner - Sifir VRAM Modu Baslatiliyor...
set WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--disable-gpu --disable-gpu-compositing
set TAURI_DISABLE_GPU=1
set TAURI_FORCE_CPU=1
npm run dev
