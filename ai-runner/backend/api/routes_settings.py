"""
AI Runner — Settings & Hardware API Routes
Implements FR-601–FR-604 and hardware profile endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Literal
import ipaddress
import logging
import os
import re

from ..core.hardware_profile import get_hardware_profile, check_vram_change, HardwareProfile
from ..db import session_store
from .auth import require_api_access

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"], dependencies=[Depends(require_api_access)])

# Cache the hardware profile for change detection
_cached_profile: Optional[HardwareProfile] = None


class SettingsValues(BaseModel):
    """Typed, allow-listed settings accepted from the UI or an import file."""

    model_config = ConfigDict(extra="forbid")

    theme: Optional[Literal["dark", "light"]] = None
    language: Optional[Literal["tr", "en"]] = None
    default_model: Optional[str] = Field(default=None, max_length=500)
    default_system_prompt: Optional[str] = Field(default=None, max_length=100_000)
    model_dir: Optional[str] = Field(default=None, max_length=2_048)
    cache_size_limit_gb: Optional[int] = Field(default=None, ge=1, le=10_000)
    n_threads: Optional[int] = Field(default=None, ge=1, le=1_024)
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    n_batch: Optional[int] = Field(default=None, ge=1, le=65_536)
    api_host: Optional[str] = Field(default=None, min_length=1, max_length=255)
    api_port: Optional[int] = Field(default=None, ge=1_024, le=65_535)
    api_key: Optional[str] = Field(default=None, max_length=512)
    allow_network_access: Optional[bool] = None
    advanced_mode: Optional[bool] = None
    kv_cache_type: Optional[Literal["q4_0", "q5_0", "q5_1", "q8_0", "f16"]] = None
    flash_attn: Optional[bool] = None
    cache_context_shift: Optional[bool] = None
    draft_model_path: Optional[str] = Field(default=None, max_length=2_048)
    draft_n_gpu_layers: Optional[int] = Field(default=None, ge=-1, le=10_000)
    speculative_decoding: Optional[bool] = None
    draft_num_pred_tokens: Optional[int] = Field(default=None, ge=1, le=64)
    max_context_length: Optional[int] = Field(default=None, ge=512, le=1_048_576)
    max_history_messages: Optional[int] = Field(default=None, ge=0, le=100_000)
    auto_context_prune: Optional[bool] = None
    context_compaction_mode: Optional[Literal["drop_oldest", "extractive_summary"]] = None
    selected_gpu_index: Optional[int] = Field(default=None, ge=0, le=128)
    tensor_split: Optional[list[float]] = Field(default=None, max_length=128)
    extreme_mode_enabled: Optional[bool] = None
    extreme_preset: Optional[Literal["safe", "balanced", "performance", "maximum_capacity"]] = None
    adaptive_load: Optional[bool] = None
    adaptive_max_attempts: Optional[int] = Field(default=None, ge=1, le=12)
    backend_preference: Optional[Literal["auto", "cuda", "vulkan", "metal", "sycl", "cpu"]] = None
    generation_timeout_s: Optional[float] = Field(default=None, ge=0.0, le=86_400.0)

    @field_validator("api_host")
    @classmethod
    def validate_api_host(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        candidate = value[1:-1] if value.startswith("[") and value.endswith("]") else value
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            hostname = candidate.rstrip(".")
            labels = hostname.split(".") if hostname else []
            if (
                len(hostname) > 253
                or not labels
                or any(
                    len(label) > 63
                    or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                    for label in labels
                )
            ):
                raise ValueError("api_host yalnızca bir IP adresi veya ana makine adı olmalıdır")
        return value

    @field_validator("api_key")
    @classmethod
    def normalise_api_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("tensor_split")
    @classmethod
    def validate_tensor_split(cls, value: Optional[list[float]]) -> Optional[list[float]]:
        if value is None:
            return None
        if not value or any(part <= 0 for part in value):
            raise ValueError("tensor_split pozitif oranlardan oluşmalıdır")
        total = sum(value)
        if total <= 0:
            raise ValueError("tensor_split toplamı sıfırdan büyük olmalıdır")
        return [round(part / total, 6) for part in value]

class SettingsUpdate(BaseModel):
    settings: SettingsValues


async def _validated_settings_patch(values: SettingsValues) -> dict:
    patch = values.model_dump(exclude_unset=True)
    merged = {**await session_store.get_settings(), **patch}
    host = str(merged.get("api_host") or "127.0.0.1").strip().lower()
    loopback = {"127.0.0.1", "localhost", "::1", "[::1]"}
    if host not in loopback and merged.get("allow_network_access") is not True:
        raise HTTPException(
            status_code=422,
            detail="Ağ erişimi için açık kullanıcı onayı gereklidir.",
        )
    configured_api_key = os.environ.get("AI_RUNNER_API_KEY", "").strip() or str(
        merged.get("api_key") or ""
    ).strip()
    if host not in loopback and not configured_api_key:
        raise HTTPException(
            status_code=422,
            detail="Ağ erişimi açılmadan önce bir API anahtarı belirlenmelidir.",
        )
    return patch


# ── Hardware Endpoints ──

@router.get("/api/hardware/profile")
async def hardware_profile():
    """FR-201–FR-203: Get current hardware profile."""
    global _cached_profile
    try:
        model_dir = await session_store.get_setting("model_dir")
        selected_gpu = int(await session_store.get_setting("selected_gpu_index", 0))
        profile = get_hardware_profile(model_dir=model_dir, selected_gpu=selected_gpu)

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
        stored_api_key = settings.pop("api_key", None)
        if os.environ.get("AI_RUNNER_API_KEY", "").strip():
            api_key_source = "environment"
        elif str(stored_api_key or "").strip():
            api_key_source = "settings"
        else:
            api_key_source = "disabled"
        return {
            "settings": {**settings, "api_key": None},
            "api_key_source": api_key_source,
            "api_key_configured": bool(
                os.environ.get("AI_RUNNER_API_KEY", "").strip()
                or str(stored_api_key or "").strip()
            ),
        }
    except Exception as e:
        logger.error(f"Settings read error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/settings")
async def update_settings(request: SettingsUpdate):
    """Update settings."""
    try:
        patch = await _validated_settings_patch(request.settings)
        await session_store.update_settings(patch)
        if patch.get("model_dir"):
            from ..models.model_manager import model_manager
            model_manager.set_model_dir(patch["model_dir"])
        return {"status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/settings/export")
async def export_settings():
    """FR-604: Export settings as JSON."""
    settings = await session_store.get_settings()
    # Secrets must never be copied into an export file.
    settings.pop("api_key", None)
    return settings


@router.post("/api/settings/import")
async def import_settings(request: SettingsUpdate):
    """FR-604: Import settings from JSON."""
    patch = await _validated_settings_patch(request.settings)
    await session_store.update_settings(patch)
    if patch.get("model_dir"):
        from ..models.model_manager import model_manager
        model_manager.set_model_dir(patch["model_dir"])
    return {"status": "imported"}
