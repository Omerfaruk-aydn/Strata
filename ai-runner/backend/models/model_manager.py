"""
AI Runner — Model Manager
HuggingFace Hub integration, download management, local library.
Implements FR-101 through FR-106.
"""

import os
import json
import asyncio
import hashlib
import re
import shutil
import time
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field
import logging

from ..core.model_loader import GGUFMetadata, validate_gguf_file

logger = logging.getLogger(__name__)


# ── Data Models ──

class ModelMetadata(BaseModel):
    """Model metadata (Section 9)."""
    id: str
    display_name: str
    parameter_count: int = 0
    available_quants: List[str] = Field(default_factory=list)
    license: str = ""
    context_length: int = 4096
    downloaded_quant: Optional[str] = None
    file_size_bytes: int = 0
    local_path: Optional[str] = None
    last_used: Optional[str] = None
    downloads: int = 0
    author: str = ""
    description: str = ""
    architecture: str = ""
    total_layers: int = 0
    metadata_valid: bool = False


class DownloadProgress(BaseModel):
    """Download progress update."""
    model_id: str
    quant: str
    progress: float  # 0.0 to 1.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_mbps: float = 0.0
    eta_seconds: int = 0
    status: str = "downloading"  # downloading, paused, completed, error


class ModelManager:
    """
    Manages model search, download, and local library.
    FR-101: HuggingFace search
    FR-102: Resumable downloads
    FR-103: Compatibility badges
    FR-104: Pre-download estimation
    FR-105: Local library management
    FR-106: Metadata caching
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(
            os.path.expanduser("~"), ".ai-runner", "models"
        )
        self._cache_dir = os.path.join(
            os.path.expanduser("~"), ".ai-runner", "cache"
        )
        self._downloads: Dict[str, asyncio.Task] = {}
        self._download_cancel: Dict[str, bool] = {}

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self._cache_dir, exist_ok=True)

    def set_model_dir(self, model_dir: str) -> None:
        """Apply a validated model directory at runtime."""
        resolved = os.path.abspath(os.path.expanduser(model_dir))
        os.makedirs(resolved, exist_ok=True)
        self.model_dir = resolved

    async def search_models(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ModelMetadata]:
        """Search Hugging Face without blocking FastAPI's event loop."""
        return await asyncio.to_thread(self._search_models_sync, query, limit)

    def _search_models_sync(
        self,
        query: str,
        limit: int = 20,
    ) -> List[ModelMetadata]:
        """
        FR-101: Search HuggingFace Hub for GGUF models.
        Returns models with name, size, downloads, and license.
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            results = []

            # Search for GGUF models
            models = api.list_models(
                filter="gguf",
                search=query,
                sort="downloads",
                limit=limit,
                full=True,
                cardData=True,
            )

            for model in models:
                # Check if this repo has GGUF files
                model_id = model.id or ""

                # Heuristic: GGUF models often have "GGUF" in the name or tags
                tags = model.tags or []
                has_gguf = (
                    "gguf" in model_id.lower() or
                    "gguf" in " ".join(tags).lower()
                )

                if not has_gguf and query.lower() != "gguf":
                    # For non-GGUF-specific searches, still include but mark
                    continue

                # Try to extract parameter count from name or tags
                param_count = self._extract_param_count(model_id, tags)

                # Get available quants from the repository's actual GGUF files.
                sibling_names = [
                    getattr(sibling, "rfilename", "")
                    for sibling in (model.siblings or [])
                ]
                quants = self._extract_quants_from_files(sibling_names)

                display_name = model_id.split("/")[-1] if "/" in model_id else model_id
                display_name = display_name.replace("-GGUF", "").replace("-gguf", "")

                card_data = model.card_data
                if hasattr(card_data, "to_dict"):
                    card_data = card_data.to_dict()
                if not isinstance(card_data, dict):
                    card_data = {}

                results.append(ModelMetadata(
                    id=model_id,
                    display_name=display_name,
                    parameter_count=param_count,
                    available_quants=quants if quants else ["Q4_K_M"],
                    license=card_data.get("license", "") or "",
                    context_length=4096,
                    downloads=model.downloads or 0,
                    author=model_id.split("/")[0] if "/" in model_id else "",
                    description=str(card_data.get("description", "") or "")[:200],
                ))

            return results

        except ImportError as exc:
            raise RuntimeError(
                "Model araması için huggingface-hub paketi gerekli."
            ) from exc
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise RuntimeError(f"Hugging Face model araması başarısız: {e}") from e

    async def download_model(
        self,
        model_id: str,
        quant: str = "Q4_K_M",
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        max_file_bytes: Optional[int] = None,
    ) -> str:
        """
        FR-102: Download a model with resumable support.
        Returns the local path of the downloaded file.
        """
        current_task = asyncio.current_task()
        safe_name = self._safe_filename_part(model_id)
        safe_quant = self._safe_filename_part(quant.upper(), max_length=40)
        identity = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
        filename = f"{safe_name}-{identity}-{safe_quant}.gguf"
        local_path = os.path.join(self.model_dir, filename)
        part_path = f"{local_path}.part"

        if os.path.exists(local_path):
            validation = await asyncio.to_thread(validate_gguf_file, local_path)
            if not validation.is_valid:
                raise RuntimeError(
                    f"Mevcut model dosyası geçerli bir GGUF değil: {validation.error or 'yapısal hata'}"
                )
            if not self._load_model_cache(filename):
                self._save_model_cache(
                    model_id,
                    quant,
                    local_path,
                    gguf_metadata=validation,
                )
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_id=model_id,
                    quant=quant,
                    progress=1.0,
                    status="completed",
                ))
            return local_path

        active = self._downloads.get(model_id)
        if active and active is not current_task and not active.done():
            raise RuntimeError("Bu model için zaten etkin bir indirme var.")
        if current_task:
            self._downloads[model_id] = current_task

        self._download_cancel[model_id] = False
        try:
            remote_filename, download_url, token = await asyncio.to_thread(
                self._resolve_download,
                model_id,
                quant,
            )
            result = await self._download_to_part(
                model_id=model_id,
                quant=quant,
                url=download_url,
                token=token,
                part_path=part_path,
                progress_callback=progress_callback,
                max_file_bytes=max_file_bytes,
            )
            if not result:
                return ""

            validation = await asyncio.to_thread(validate_gguf_file, part_path)
            if not validation.is_valid:
                raise RuntimeError(
                    f"İndirilen GGUF yapısal doğrulamadan geçemedi: {validation.error or 'bilinmeyen hata'}"
                )

            os.replace(part_path, local_path)
            checksum = await asyncio.to_thread(self._sha256, local_path)
            self._save_model_cache(
                model_id,
                quant,
                local_path,
                checksum=checksum,
                remote_filename=remote_filename,
                gguf_metadata=validation,
            )

            if progress_callback:
                progress_callback(DownloadProgress(
                    model_id=model_id,
                    quant=quant,
                    progress=1.0,
                    status="completed",
                ))

            return local_path

        except Exception as e:
            logger.error(f"Download error: {e}")
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_id=model_id,
                    quant=quant,
                    progress=0.0,
                    downloaded_bytes=os.path.getsize(part_path) if os.path.exists(part_path) else 0,
                    status="error",
                ))
            raise
        finally:
            if self._downloads.get(model_id) is current_task:
                self._downloads.pop(model_id, None)

    async def _download_to_part(
        self,
        model_id: str,
        quant: str,
        url: str,
        token: Optional[str],
        part_path: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]],
        max_file_bytes: Optional[int],
    ) -> str:
        """Stream a Hugging Face file with HTTP Range resume support."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Model indirmek için httpx paketi gerekli.") from exc

        resume_from = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        headers: Dict[str, str] = {}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        if token:
            headers["Authorization"] = f"Bearer {token}"

        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 416:
                    content_range = response.headers.get("content-range", "")
                    total_match = re.search(r"\*/(\d+)$", content_range)
                    if (
                        total_match
                        and os.path.getsize(part_path) == int(total_match.group(1))
                        and self._is_valid_gguf(part_path)
                    ):
                        return part_path
                    raise RuntimeError("Kısmi GGUF dosyası uzak dosyayla eşleşmiyor; yeniden indirin.")
                response.raise_for_status()

                append = response.status_code == 206 and resume_from > 0
                if not append:
                    resume_from = 0

                content_range = response.headers.get("content-range", "")
                if append:
                    start_match = re.match(r"bytes\s+(\d+)-\d+/(?:\d+|\*)", content_range)
                    if not start_match or int(start_match.group(1)) != resume_from:
                        raise RuntimeError("Sunucu geçersiz bir Range yanıtı döndürdü.")
                range_match = re.search(r"/(\d+)$", content_range)
                if range_match:
                    total_bytes = int(range_match.group(1))
                else:
                    content_length = int(response.headers.get("content-length", "0") or 0)
                    total_bytes = resume_from + content_length if content_length else 0

                if total_bytes and max_file_bytes is not None and total_bytes > max_file_bytes:
                    raise RuntimeError("Model, ayarlanan önbellek boyutu sınırını aşıyor.")
                required_bytes = max(total_bytes - resume_from, 0)
                free_bytes = shutil.disk_usage(os.path.dirname(part_path)).free
                reserve_bytes = 512 * 1024 * 1024
                if required_bytes and required_bytes > max(free_bytes - reserve_bytes, 0):
                    raise RuntimeError("Model indirmesi için yeterli boş disk alanı yok.")

                downloaded = resume_from
                started_at = time.monotonic()
                last_report_at = 0.0
                mode = "ab" if append else "wb"

                with open(part_path, mode) as output:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if self._download_cancel.get(model_id, False):
                            if progress_callback:
                                progress_callback(DownloadProgress(
                                    model_id=model_id,
                                    quant=quant,
                                    progress=(downloaded / total_bytes) if total_bytes else 0.0,
                                    downloaded_bytes=downloaded,
                                    total_bytes=total_bytes,
                                    status="paused",
                                ))
                            return ""

                        if max_file_bytes is not None and downloaded + len(chunk) > max_file_bytes:
                            raise RuntimeError("Model, ayarlanan önbellek boyutu sınırını aşıyor.")
                        output.write(chunk)
                        downloaded += len(chunk)
                        elapsed = max(time.monotonic() - started_at, 0.001)
                        now = time.monotonic()
                        if progress_callback and (now - last_report_at >= 0.25):
                            transferred = max(downloaded - resume_from, 0)
                            bytes_per_second = transferred / elapsed
                            remaining = max(total_bytes - downloaded, 0)
                            progress_callback(DownloadProgress(
                                model_id=model_id,
                                quant=quant,
                                progress=(downloaded / total_bytes) if total_bytes else 0.0,
                                downloaded_bytes=downloaded,
                                total_bytes=total_bytes,
                                speed_mbps=bytes_per_second / (1024 * 1024),
                                eta_seconds=int(remaining / bytes_per_second) if bytes_per_second > 0 else 0,
                                status="downloading",
                            ))
                            last_report_at = now

        return part_path

    def cancel_download(self, model_id: str) -> bool:
        """Pause/cancel an active download."""
        if model_id not in self._downloads:
            return False
        self._download_cancel[model_id] = True
        return True

    def get_local_models(self) -> List[ModelMetadata]:
        """
        FR-105: List all locally downloaded models.
        Returns models with size, last used date, and disk usage.
        """
        models = []
        cache = self._load_all_caches()

        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.gguf'):
                continue

            filepath = os.path.join(self.model_dir, filename)
            stat = os.stat(filepath)
            file_size = stat.st_size

            # Check cache for metadata
            cached = cache.get(filename, {})
            metadata_cached = (
                cached.get("metadata_parser_version") == 2
                and cached.get("metadata_mtime_ns") == stat.st_mtime_ns
                and cached.get("metadata_file_size") == file_size
                and cached.get("metadata_valid") is True
            )
            if metadata_cached:
                if not self._has_gguf_magic(filepath):
                    logger.warning("Ignoring invalid GGUF file: %s", filepath)
                    continue
                gguf = None
            else:
                gguf = validate_gguf_file(filepath)
                if not gguf.is_valid:
                    logger.warning("Ignoring invalid GGUF file %s: %s", filepath, gguf.error)
                    continue
                cached.update({
                    "metadata_valid": True,
                    "metadata_parser_version": 2,
                    "metadata_mtime_ns": stat.st_mtime_ns,
                    "metadata_file_size": file_size,
                    "parameter_count": gguf.parameter_count or cached.get("parameter_count", 0),
                    "context_length": gguf.context_length,
                    "architecture": gguf.architecture,
                    "total_layers": gguf.block_count,
                })
                self._save_cache_file(filename, cached)

            model_id = cached.get("model_id", filename.replace(".gguf", ""))
            quant = cached.get("quant") or self._extract_quant_from_filename(filename) or "Q4_K_M"
            display_name = cached.get("display_name", filename.replace(".gguf", ""))
            parameter_count = int(cached.get("parameter_count", 0) or (gguf.parameter_count if gguf else 0) or 0)

            models.append(ModelMetadata(
                id=model_id,
                display_name=display_name,
                parameter_count=parameter_count,
                available_quants=[quant],
                downloaded_quant=quant,
                file_size_bytes=file_size,
                local_path=filepath,
                last_used=cached.get("last_used"),
                license=cached.get("license", ""),
                context_length=int(cached.get("context_length", 0) or (gguf.context_length if gguf else 0) or 4096),
                architecture=str(cached.get("architecture", "") or (gguf.architecture if gguf else "")),
                total_layers=int(cached.get("total_layers", 0) or (gguf.block_count if gguf else 0)),
                metadata_valid=bool(cached.get("metadata_valid") or (gguf.is_valid if gguf else False)),
            ))

        return models

    def delete_model(self, model_id: str, quant: Optional[str] = None) -> bool:
        """FR-105: Delete a locally downloaded model."""
        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.gguf'):
                continue
            filepath = os.path.join(self.model_dir, filename)
            cached = self._load_model_cache(filename)
            inferred_id = filename.removesuffix(".gguf")
            cached_id = cached.get("model_id")
            cached_quant = cached.get("quant", "Q4_K_M")
            id_matches = cached_id == model_id or (not cached and inferred_id == model_id)
            quant_matches = quant is None or cached_quant == quant
            if id_matches and quant_matches:
                try:
                    os.remove(filepath)
                    # Remove cache file too
                    cache_path = os.path.join(
                        self._cache_dir, f"{filename}.json"
                    )
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                    logger.info(f"Deleted model: {model_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete model: {e}")
                    return False
        return False

    def update_last_used(self, model_id: str) -> None:
        """Update the last_used timestamp for a model."""
        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.gguf'):
                continue
            cached = self._load_model_cache(filename)
            if cached.get("model_id") == model_id:
                cached["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._save_cache_file(filename, cached)
                break

    def register_local_model(
        self,
        model_id: str,
        quant: str,
        local_path: str,
        *,
        source: str = "local",
        compute_checksum: bool = False,
    ) -> None:
        """Register a validated local/quantized GGUF in the offline cache."""
        resolved = os.path.abspath(local_path)
        model_root = os.path.abspath(self.model_dir)
        if os.path.commonpath([resolved, model_root]) != model_root:
            raise ValueError("Model dosyası yapılandırılmış model klasöründe olmalıdır.")
        validation = validate_gguf_file(resolved)
        if not validation.is_valid:
            raise ValueError(f"Geçerli bir GGUF dosyası gerekli: {validation.error or 'yapısal hata'}")
        checksum = self._sha256(resolved) if compute_checksum else None
        self._save_model_cache(
            model_id,
            quant.upper(),
            resolved,
            checksum=checksum,
            gguf_metadata=validation,
        )
        filename = os.path.basename(resolved)
        cached = self._load_model_cache(filename)
        cached["source"] = source
        self._save_cache_file(filename, cached)

    def get_compatibility_badge(
        self,
        model: ModelMetadata,
        vram_free_mb: int,
        ram_free_mb: int,
    ) -> str:
        """
        FR-103: Calculate compatibility badge.
        🟢 Rahat çalışır / 🟡 Kısıtlı çalışır / 🔴 Önerilmez
        """
        file_size_mb = model.file_size_bytes / (1024 * 1024) if model.file_size_bytes > 0 else 0

        if file_size_mb == 0:
            # Estimate from parameter count and default quant
            from ..core.memory_manager import estimate_model_size_mb
            file_size_mb = estimate_model_size_mb(
                model.parameter_count,
                model.downloaded_quant or "Q4_K_M"
            )

        usable_vram = vram_free_mb * 0.85
        usable_ram = ram_free_mb * 0.80
        total_usable = usable_vram + usable_ram

        if file_size_mb <= usable_vram:
            return "compatible"  # 🟢 Fits entirely in VRAM
        elif file_size_mb <= total_usable:
            return "limited"    # 🟡 Needs offload to RAM
        else:
            return "incompatible"  # 🔴 Doesn't fit

    # ── Cache Management (FR-106) ──

    def _save_model_cache(
        self,
        model_id: str,
        quant: str,
        local_path: str,
        checksum: Optional[str] = None,
        remote_filename: Optional[str] = None,
        gguf_metadata: Optional[GGUFMetadata] = None,
    ) -> None:
        """Save model metadata to cache for offline access."""
        filename = os.path.basename(local_path)
        cache_data = {
            "model_id": model_id,
            "quant": quant,
            "display_name": model_id.split("/")[-1].replace("-GGUF", ""),
            "local_path": local_path,
            "parameter_count": self._extract_param_count(model_id, []),
            "sha256": checksum,
            "remote_filename": remote_filename,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        gguf = gguf_metadata or validate_gguf_file(local_path)
        if gguf.is_valid:
            stat = os.stat(local_path)
            cache_data.update({
                "parameter_count": gguf.parameter_count or cache_data["parameter_count"],
                "context_length": gguf.context_length,
                "architecture": gguf.architecture,
                "total_layers": gguf.block_count,
                "metadata_valid": True,
                "metadata_parser_version": 2,
                "metadata_mtime_ns": stat.st_mtime_ns,
                "metadata_file_size": stat.st_size,
            })
        self._save_cache_file(filename, cache_data)

    def _save_cache_file(self, filename: str, data: dict) -> None:
        cache_path = os.path.join(self._cache_dir, f"{filename}.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Cache write error: {e}")

    def _load_model_cache(self, filename: str) -> dict:
        cache_path = os.path.join(self._cache_dir, f"{filename}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _load_all_caches(self) -> dict:
        caches = {}
        if os.path.exists(self._cache_dir):
            for cache_file in os.listdir(self._cache_dir):
                if cache_file.endswith('.json'):
                    gguf_name = cache_file.replace('.json', '')
                    caches[gguf_name] = self._load_model_cache(gguf_name)
        return caches

    # ── Helper Methods ──

    @staticmethod
    def _safe_filename_part(value: str, max_length: int = 120) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
        if not cleaned:
            raise ValueError("Geçersiz model veya quant adı.")
        return cleaned[:max_length]

    @staticmethod
    def _is_valid_gguf(filepath: str) -> bool:
        return validate_gguf_file(filepath).is_valid

    @staticmethod
    def _has_gguf_magic(filepath: str) -> bool:
        try:
            with open(filepath, "rb") as model_file:
                return model_file.read(4) == b"GGUF"
        except OSError:
            return False

    @staticmethod
    def _sha256(filepath: str) -> str:
        digest = hashlib.sha256()
        with open(filepath, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_download(self, model_id: str, quant: str) -> tuple[str, str, Optional[str]]:
        try:
            from huggingface_hub import HfApi, get_token, hf_hub_url
        except ImportError as exc:
            raise RuntimeError("Model indirmek için huggingface-hub paketi gerekli.") from exc

        files = HfApi().list_repo_files(repo_id=model_id, repo_type="model")
        gguf_files = [name for name in files if name.lower().endswith(".gguf")]
        quant_key = re.sub(r"[^a-z0-9]+", "_", quant.lower()).strip("_")
        candidates = [
            name for name in gguf_files
            if re.search(
                rf"(?:^|_){re.escape(quant_key)}(?:_|$)",
                re.sub(r"[^a-z0-9]+", "_", name.lower()),
            )
        ]
        if not candidates:
            available = ", ".join(os.path.basename(name) for name in gguf_files[:8])
            raise FileNotFoundError(
                f"{quant} quant dosyası depoda bulunamadı. "
                f"Mevcut GGUF dosyaları: {available or 'yok'}"
            )

        non_split = [
            name for name in candidates
            if not re.search(r"-\d{5}-of-\d{5}\.gguf$", name, re.I)
        ]
        if not non_split:
            raise RuntimeError(
                "Çok parçalı GGUF indirme henüz desteklenmiyor; tek dosyalı bir quant seçin."
            )
        remote_filename = sorted(non_split, key=lambda name: (len(name), name.lower()))[0]
        return remote_filename, hf_hub_url(model_id, remote_filename), get_token()

    def _extract_param_count(self, model_id: str, tags: List[str]) -> int:
        """Heuristic to extract parameter count from model name."""
        source = " ".join([model_id, *tags]).lower()
        match = re.search(
            r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])",
            source,
        )
        if match:
            return int(float(match.group(1)) * 1_000_000_000)
        return 0

    def _extract_quants_from_files(self, filenames: List[str]) -> List[str]:
        """Extract quantization levels from actual GGUF repository filenames."""
        known_quants = [
            "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XS", "IQ3_S", "IQ4_XS", "IQ4_NL",
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16", "BF16",
        ]
        found = set()
        for filename in filenames:
            if not filename.lower().endswith(".gguf"):
                continue
            normalized = re.sub(r"[^a-z0-9]+", "_", filename.lower())
            for quant in known_quants:
                key = quant.lower()
                if re.search(rf"(?:^|_){re.escape(key)}(?:_|$)", normalized):
                    found.add(quant)
        return [quant for quant in known_quants if quant in found]

    def _extract_quant_from_filename(self, filename: str) -> Optional[str]:
        quants = self._extract_quants_from_files([filename])
        return quants[0] if quants else None

# Singleton instance
model_manager = ModelManager()
