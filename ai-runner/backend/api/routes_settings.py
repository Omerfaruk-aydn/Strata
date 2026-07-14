"""
AI Runner — Settings & Hardware API Routes
Implements FR-601–FR-604 and hardware profile endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import logging

from ..core.hardware_profile import get_hardware_profile, check_vram_change, HardwareProfile
from ..db import session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])

# Cache the hardware profile for change detection
_cached_profile: Optional[HardwareProfile] = None


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]


# ── Hardware Endpoints ──

@router.get("/api/hardware/profile")
async def hardware_profile():
    """FR-201–FR-203: Get current hardware profile."""
    global _cached_profile
    try:
        model_dir = await session_store.get_setting("model_dir")
        profile = get_hardware_profile(model_dir=model_dir)

        # Check for VRAM changes (FR-204)
        warning = None
        if _cached_profile:
            warning = check_vram_change(_cached_profile)

        _cached_profile = profile

        result = profile.model_dump()
        if warning:
            result["vram_warning"] = warning

        return result

    except Exception as e:
        logger.error(f"Hardware profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/hardware/refresh")
async def refresh_hardware():
    """FR-203: Manually refresh hardware profile."""
    global _cached_profile
    _cached_profile = None
    return await hardware_profile()


# ── Settings Endpoints (FR-601–FR-604) ──

@router.get("/api/settings")
async def get_settings():
    """Get all settings."""
    try:
        settings = await session_store.get_settings()
        return {"settings": settings}
    except Exception as e:
        logger.error(f"Settings read error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings")
async def update_settings(request: SettingsUpdate):
    """Update settings."""
    try:
        await session_store.update_settings(request.settings)
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/export")
async def export_settings():
    """FR-604: Export settings as JSON."""
    settings = await session_store.get_settings()
    return settings


@router.post("/api/settings/import")
async def import_settings(request: SettingsUpdate):
    """FR-604: Import settings from JSON."""
    await session_store.update_settings(request.settings)
    return {"status": "imported"}
