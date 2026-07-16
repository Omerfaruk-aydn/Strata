"""Managed llama.cpp quantization jobs for locally installed GGUF models."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .model_loader import validate_gguf_file


SUPPORTED_OUTPUT_QUANTS = [
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XS", "IQ3_S", "IQ4_XS", "IQ4_NL",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
]


class QuantizationJob(BaseModel):
    id: str
    model_id: str
    source_path: str
    output_path: str
    output_quant: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    allow_requantize: bool = False


class QuantizationManager:
    """Runs one bounded external quantization process at a time."""

    def __init__(self) -> None:
        self._jobs: Dict[str, QuantizationJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._active_job_id: Optional[str] = None
        self._lock = asyncio.Lock()

    def list_jobs(self) -> List[QuantizationJob]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get_job(self, job_id: str) -> Optional[QuantizationJob]:
        return self._jobs.get(job_id)

    async def start_job(
        self,
        *,
        executable: str,
        model_id: str,
        source_path: str,
        model_dir: str,
        output_quant: str,
        threads: int,
        allow_requantize: bool,
    ) -> QuantizationJob:
        quant = output_quant.upper()
        if quant not in SUPPORTED_OUTPUT_QUANTS:
            raise ValueError(f"Unsupported output quantization: {quant}")
        executable_path = Path(executable).resolve()
        if not executable_path.is_file():
            raise FileNotFoundError("llama-quantize executable was not found")

        source = Path(source_path).resolve()
        root = Path(model_dir).resolve()
        if not source.is_file() or source.suffix.lower() != ".gguf":
            raise FileNotFoundError("Source GGUF file was not found")
        if root not in source.parents:
            raise ValueError("Source model must be inside the configured model directory")
        source_validation = await asyncio.to_thread(validate_gguf_file, str(source))
        if not source_validation.is_valid:
            raise ValueError(
                f"Source GGUF failed structural validation: {source_validation.error or 'unknown error'}"
            )

        safe_quant = re.sub(r"[^A-Z0-9_]+", "_", quant)
        output = source.with_name(f"{source.stem}-{safe_quant}.gguf")
        if output.exists():
            raise FileExistsError(f"Output model already exists: {output.name}")

        async with self._lock:
            if self._active_job_id:
                active = self._jobs.get(self._active_job_id)
                if active and active.status in {"queued", "running"}:
                    raise RuntimeError("Another quantization job is already running")
            job = QuantizationJob(
                id=f"quant_{uuid.uuid4().hex}",
                model_id=model_id,
                source_path=str(source),
                output_path=str(output),
                output_quant=quant,
                allow_requantize=allow_requantize,
            )
            self._jobs[job.id] = job
            self._active_job_id = job.id
            self._tasks[job.id] = asyncio.create_task(
                self._run_job(job.id, str(executable_path), max(1, min(int(threads), 1024)))
            )
            return job

    async def _run_job(self, job_id: str, executable: str, threads: int) -> None:
        job = self._jobs[job_id]
        command = [executable]
        if job.allow_requantize:
            command.append("--allow-requantize")
        command.extend([job.source_path, job.output_path, job.output_quant, str(threads)])
        try:
            job.status = "running"
            job.started_at = time.time()
            job.progress = 0.02
            job.message = "Quantization process started"
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            self._processes[job_id] = process
            assert process.stdout is not None
            async for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    job.message = line[-500:]
                    parsed = self._parse_progress(line)
                    if parsed is not None:
                        job.progress = max(job.progress, min(parsed, 0.98))
            return_code = await process.wait()
            job.return_code = return_code
            if job.status == "cancelled":
                return
            if return_code != 0:
                raise RuntimeError(job.message or f"llama-quantize exited with code {return_code}")
            validation = await asyncio.to_thread(validate_gguf_file, job.output_path)
            if not validation.is_valid:
                raise RuntimeError(
                    "Quantized output failed structural validation: "
                    f"{validation.error or 'unknown error'}"
                )
            job.status = "completed"
            job.progress = 1.0
            job.message = "Quantization completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.message = "Quantization cancelled"
            self._remove_partial_output(job)
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)[:2000]
            job.message = "Quantization failed"
            self._remove_partial_output(job)
        finally:
            job.finished_at = time.time()
            self._processes.pop(job_id, None)
            self._tasks.pop(job_id, None)
            if self._active_job_id == job_id:
                self._active_job_id = None

    @staticmethod
    def _parse_progress(line: str) -> Optional[float]:
        match = re.search(r"(?:^|\s)(\d{1,3}(?:\.\d+)?)\s*%", line)
        if match:
            return float(match.group(1)) / 100
        lowered = line.lower()
        if "loading model" in lowered:
            return 0.08
        if "quantizing" in lowered:
            return 0.25
        if "writing" in lowered or "saving" in lowered:
            return 0.9
        return None

    async def cancel_job(self, job_id: str) -> QuantizationJob:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status not in {"queued", "running"}:
            return job
        job.status = "cancelled"
        job.message = "Cancellation requested"
        process = self._processes.get(job_id)
        if process and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        self._remove_partial_output(job)
        job.finished_at = time.time()
        return job

    async def shutdown(self) -> None:
        """Terminate active native jobs so they cannot outlive the backend."""
        tasks = [task for task in self._tasks.values() if not task.done()]
        active_ids = [
            job.id for job in self._jobs.values()
            if job.status in {"queued", "running"}
        ]
        for job_id in active_ids:
            try:
                await self.cancel_job(job_id)
            except Exception:
                pass
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _remove_partial_output(job: QuantizationJob) -> None:
        try:
            output = Path(job.output_path)
            if output.exists():
                output.unlink()
        except OSError:
            pass


quantization_manager = QuantizationManager()
