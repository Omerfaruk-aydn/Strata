"""
AI Runner — Backend Entry Point
FastAPI application with CORS, routers, and database initialization.
"""

import os
import sys

# --- Dynamic CUDA DLL directory loader for Windows ---
if sys.platform == "win32":
    for path in sys.path:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.isdir(nvidia_path):
            for sub in os.listdir(nvidia_path):
                bin_path = os.path.join(nvidia_path, sub, "bin")
                if os.path.isdir(bin_path):
                    try:
                        os.add_dll_directory(bin_path)
                        os.environ["PATH"] = bin_path + os.path.pathsep + os.environ["PATH"]
                    except Exception:
                        pass

import logging
import logging.handlers
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Logging Setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log_dir = os.path.join(os.path.expanduser("~"), ".ai-runner", "logs")
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(log_dir, "backend.log"),
    maxBytes=10 * 1024 * 1024,  # 10MB per file
    backupCount=5,              # Keep 5 rotated files
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

# ── Import Routers ──
from .api.routes_models import router as models_router
from .api.routes_chat import router as chat_router
from .api.routes_settings import router as settings_router
from .api.ws_telemetry import router as telemetry_router
from .api.routes_optimizer import router as optimizer_router
from .api.routes_extreme import router as extreme_router
from .api.routes_ultra import router as ultra_router
from .db import session_store
from .api.auth import TRUSTED_BROWSER_ORIGINS, require_api_access


# ── Application Lifecycle ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    logger.info("AI Runner backend starting...")
    await session_store.init_db()
    await session_store.ensure_default_settings()
    from .models.model_manager import model_manager
    model_dir = await session_store.get_setting("model_dir")
    if model_dir:
        model_manager.set_model_dir(model_dir)
    logger.info("Database initialized")
    yield
    logger.info("AI Runner backend shutting down...")
    from .core.quantization_service import quantization_manager
    from .core.inference_engine import engine
    await quantization_manager.shutdown()
    engine.unload_model()
    logger.info("Shutdown complete")


# ── FastAPI App ──

app = FastAPI(
    title="AI Runner",
    description="Yerel LLM Çalıştırma Platformu API'si",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(TRUSTED_BROWSER_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(chat_router)
app.include_router(settings_router)
app.include_router(telemetry_router)
app.include_router(optimizer_router)
app.include_router(extreme_router)
app.include_router(ultra_router)


@app.get("/")
async def root():
    return {"name": "AI Runner", "version": "1.0.0", "status": "running"}


@app.get("/api/status", dependencies=[Depends(require_api_access)])
async def status():
    from .core.inference_engine import engine
    from .core.hardware_profile import get_hardware_profile
    profile = get_hardware_profile()
    return {
        "status": "running",
        "model_loaded": engine.is_loaded,
        "model_id": engine.model_info.model_id if engine.model_info else None,
        "gpu": profile.gpu.name,
        "vram_free_mb": profile.gpu.vram_free_mb,
        "ram_free_mb": profile.ram.free_mb,
    }


if __name__ == "__main__":
    from .cli import run
    run()
