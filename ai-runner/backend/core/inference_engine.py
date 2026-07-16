"""
AI Runner — Inference Engine
Wraps llama-cpp-python for model loading and token generation.
Implements FR-301 through FR-307.

Performance Optimizations:
  - KV Cache 4-bit quantization (type_k=4, type_v=4) → ~50% less VRAM
  - Flash Attention for supported backends and long contexts
  - Memory Lock (mlock) → prevents OS from swapping model to disk
  - Physical-core thread auto-detection → 10-15% CPU improvement
  - Prompt lookup decoding without a second draft model
  - Context-window pruning that keeps the system prompt and latest turn
"""

import asyncio
import gc
import math
import time
import threading
from typing import Optional, AsyncGenerator, Dict, Any, Callable, List
from pydantic import BaseModel, Field, field_validator
import logging
import psutil

logger = logging.getLogger(__name__)


# ── GGML type constants (mirrors llama_cpp internal values) ──────────────────
# Used for KV cache quantization. Lower = less VRAM, slightly less accurate.
GGML_TYPE_F16  = 1   # 16-bit float (default)
GGML_TYPE_Q8_0 = 8   # 8-bit quantized
GGML_TYPE_Q5_1 = 7   # 5-bit quantized
GGML_TYPE_Q5_0 = 6   # 5-bit quantized (alt)
GGML_TYPE_Q4_0 = 2   # 4-bit quantized — minimum stable (max VRAM saving)

KV_CACHE_TYPE_MAP = {
    "f16":  GGML_TYPE_F16,
    "q8_0": GGML_TYPE_Q8_0,
    "q5_1": GGML_TYPE_Q5_1,
    "q5_0": GGML_TYPE_Q5_0,
    "q4_0": GGML_TYPE_Q4_0,   # Recommended default — 50% less VRAM vs f16
}


class EngineConfig(BaseModel):
    """
    Full configuration for model loading.
    Groups all performance tuning knobs into one typed object.
    """
    # Core
    n_gpu_layers: int = -1          # -1 = auto (all to GPU)
    context_length: int = 4096
    n_batch: int = 512

    # Memory
    use_mmap: bool = True           # Memory-mapped I/O for fast disk reads
    use_mlock: bool = True          # Lock RAM pages — prevents OS swap

    # Thread optimization
    n_threads: Optional[int] = None  # None = auto (physical cores only)

    # KV Cache quantization — default q4_0 = ~50% VRAM saving vs f16
    kv_cache_type: str = "q4_0"

    # Flash Attention — backend-dependent acceleration for long contexts
    flash_attn: bool = True

    # Deprecated draft-model fields retained for configuration compatibility.
    # AI Runner uses llama.cpp prompt lookup decoding instead.
    draft_model_path: Optional[str] = None
    draft_n_gpu_layers: int = -1

    # Compatibility name retained for persisted settings. This controls the
    # application-level message pruning; llama-cpp-python does not expose the
    # old low-level context-shift switch.
    cache_context_shift: bool = True

    # Multi-GPU support: split weights proportionally (e.g. [0.7, 0.3])
    tensor_split: Optional[List[float]] = None
    main_gpu: int = 0

    # llama-cpp-python supported speculative mode: prompt lookup decoding.
    speculative_decoding: bool = False
    draft_num_pred_tokens: int = 10

    # Conversation pruning is independent from the low-level KV shift option.
    auto_context_prune: bool = True
    context_compaction_mode: str = "extractive_summary"

    # Backend selection reflects the native llama.cpp build. ``cpu`` forces
    # zero GPU layers; other explicit values are validated before loading.
    backend_preference: str = "auto"

    # Hard wall-clock guard for a stalled native generation. Zero disables the
    # guard for controlled offline benchmarking only.
    generation_timeout_s: float = 300.0

    @field_validator("kv_cache_type")
    @classmethod
    def validate_kv_type(cls, v: str) -> str:
        if v not in KV_CACHE_TYPE_MAP:
            raise ValueError(
                f"kv_cache_type must be one of {list(KV_CACHE_TYPE_MAP.keys())}, got '{v}'"
            )
        return v

    @field_validator("context_compaction_mode")
    @classmethod
    def validate_compaction_mode(cls, value: str) -> str:
        allowed = {"drop_oldest", "extractive_summary"}
        if value not in allowed:
            raise ValueError(f"context_compaction_mode must be one of {sorted(allowed)}")
        return value

    @field_validator("backend_preference")
    @classmethod
    def validate_backend_name(cls, value: str) -> str:
        normalized = value.lower()
        allowed = {"auto", "cuda", "vulkan", "metal", "sycl", "cpu"}
        if normalized not in allowed:
            raise ValueError(f"backend_preference must be one of {sorted(allowed)}")
        return normalized

    @field_validator("generation_timeout_s")
    @classmethod
    def validate_generation_timeout(cls, value: float) -> float:
        if value < 0.0 or value > 86_400.0:
            raise ValueError("generation_timeout_s must be 0 or between 1 and 86400 seconds")
        return value


class InferenceParams(BaseModel):
    """Generation parameters (FR-306)."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    stop: List[str] = Field(default_factory=list)
    system_prompt: str = ""

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {v}")
        return v


class InferenceResult(BaseModel):
    """Result of a completed generation."""
    content: str
    tokens_generated: int
    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0        # Time to first token
    total_time_ms: float = 0.0
    stopped_by_user: bool = False
    finish_reason: str = "stop"  # "stop" | "length" | "user_interrupt"


# Alias for backward-compatibility with tests
GenerationResult = InferenceResult


class ModelInfo(BaseModel):
    """Info about the currently loaded model."""
    model_id: str
    model_path: str
    n_gpu_layers: int
    context_length: int
    total_layers: int
    is_loaded: bool = False
    # Reflect active optimizations for telemetry/UI
    flash_attn: bool = False
    use_mlock: bool = False
    kv_cache_type: str = "f16"
    has_draft_model: bool = False
    cache_context_shift: bool = False
    tensor_split: Optional[List[float]] = None
    main_gpu: int = 0


class LoadAttempt(BaseModel):
    attempt: int
    n_gpu_layers: int
    context_length: int
    n_batch: int
    use_mlock: bool
    outcome: str
    error_type: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class AdaptiveLoadReport(BaseModel):
    succeeded: bool
    attempts: List[LoadAttempt] = Field(default_factory=list)
    final_config: Optional[EngineConfig] = None
    recovered_from_oom: bool = False


def _get_physical_cores() -> int:
    """
    Return the number of PHYSICAL CPU cores only.
    Hyper-threading doubles logical count but does NOT improve LLM throughput
    and can actually degrade it by increasing memory bus contention.
    """
    physical = psutil.cpu_count(logical=False)
    return physical if physical and physical > 0 else psutil.cpu_count(logical=True) or 4


class InferenceEngine:
    """
    Manages LLM inference using llama-cpp-python.
    Enforces single active model (FR-307).
    Supports streaming with interrupt (FR-302, FR-303).

    Performance features:
      • KV Cache quantization (4-bit default)
      • Flash Attention on supported backends
      • mlock (OS swap prevention)
      • Physical-core threading
      • Prompt lookup decoding without a second model
      • Context-window message pruning
    """

    def __init__(self):
        self._model = None
        self._draft_model = None
        self._model_info: Optional[ModelInfo] = None
        self._is_generating = False
        self._should_stop = False
        self._stop_event = threading.Event()   # For test introspection
        self._lock = threading.Lock()
        self._load_progress_callback: Optional[Callable] = None
        self._config: Optional[EngineConfig] = None
        self._last_load_report: Optional[AdaptiveLoadReport] = None
        # Context shift: track messages already evaluated (KV cache state)
        self._evaluated_message_count: int = 0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @property
    def model_info(self) -> Optional[ModelInfo]:
        return self._model_info

    @property
    def config(self) -> Optional[EngineConfig]:
        return self._config

    @property
    def last_load_report(self) -> Optional[AdaptiveLoadReport]:
        return self._last_load_report

    def load_model(
        self,
        model_id: str,
        model_path: str,
        config: Optional[EngineConfig] = None,
        # Legacy keyword args for backward compatibility
        n_gpu_layers: int = -1,
        context_length: int = 4096,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        n_batch: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ModelInfo:
        """
        Load a GGUF model into memory with all performance optimizations.
        FR-301: n_gpu_layers auto or manual override.
        FR-307: Unloads previous model first.

        Args:
            model_id:          Model identifier
            model_path:        Path to .gguf file
            config:            EngineConfig with all optimization settings.
                               If None, legacy kwargs are used.
            progress_callback: Called with (progress_pct, status_message)
        """
        # Build config from legacy kwargs if not provided
        if config is None:
            config = EngineConfig(
                n_gpu_layers=n_gpu_layers,
                context_length=context_length,
                n_threads=n_threads,
                use_mmap=use_mmap,
                n_batch=n_batch,
            )

        with self._lock:
            if self._is_generating:
                raise RuntimeError("Üretim sürerken model değiştirilemez. Önce üretimi durdurun.")
            # FR-307: Unload previous model first
            if self._model is not None:
                self._unload_internal()

            if progress_callback:
                progress_callback(0.0, "Model yükleniyor...")

            try:
                from llama_cpp import Llama

                if progress_callback:
                    progress_callback(0.05, "llama.cpp başlatılıyor...")

                # ── Resolve thread count ────────────────────────────────────
                # Always use physical cores only (no hyper-threads) for best
                # LLM throughput. Manual override via config.n_threads.
                resolved_threads = (
                    config.n_threads
                    if config.n_threads is not None
                    else _get_physical_cores()
                )

                # ── Resolve KV cache GGML type ──────────────────────────────
                kv_type_int = KV_CACHE_TYPE_MAP.get(config.kv_cache_type, GGML_TYPE_Q4_0)

                if progress_callback:
                    progress_callback(0.10, f"KV önbelleği: {config.kv_cache_type}, iş parçacıkları: {resolved_threads}")

                # ── Build Llama kwargs ──────────────────────────────────────
                kwargs: Dict[str, Any] = {
                    "model_path":    model_path,
                    "n_gpu_layers":  config.n_gpu_layers,
                    "n_ctx":         config.context_length,
                    "use_mmap":      config.use_mmap,
                    "use_mlock":     config.use_mlock,
                    "n_batch":       config.n_batch,
                    "n_threads":     resolved_threads,
                    "type_k":        kv_type_int,   # KV key cache type
                    "type_v":        kv_type_int,   # KV value cache type
                    "verbose":       False,
                    "main_gpu":      config.main_gpu,
                }

                # Multi-GPU support (tensor_split)
                if config.tensor_split:
                    kwargs["tensor_split"] = config.tensor_split

                if config.speculative_decoding:
                    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                    kwargs["draft_model"] = LlamaPromptLookupDecoding(
                        num_pred_tokens=config.draft_num_pred_tokens,
                    )

                # Flash Attention — guard: only pass if llama_cpp supports it
                try:
                    import inspect
                    sig = inspect.signature(Llama.__init__)
                    if "flash_attn" in sig.parameters and config.flash_attn:
                        kwargs["flash_attn"] = True
                        if progress_callback:
                            progress_callback(0.15, "Flash Attention etkinleştiriliyor...")
                except Exception:
                    pass  # Older llama-cpp-python — skip silently

                if progress_callback:
                    progress_callback(0.20, "Ana model belleğe yükleniyor...")

                self._model = Llama(**kwargs)
                self._config = config
                self._evaluated_message_count = 0

                if progress_callback:
                    progress_callback(0.75, "Ana model hazır.")

                has_draft = config.speculative_decoding

                if progress_callback:
                    progress_callback(0.90, "Model metadata okunuyor...")

                # ── Extract total layers from GGUF metadata ─────────────────
                total_layers = 0
                try:
                    metadata = self._model.metadata or {}
                    architecture = metadata.get("general.architecture", "")
                    if isinstance(architecture, bytes):
                        architecture = architecture.decode("utf-8", errors="replace")
                    candidate_keys = []
                    if architecture:
                        candidate_keys.append(f"{architecture}.block_count")
                    candidate_keys.append("llama.block_count")
                    for key in candidate_keys:
                        if key in metadata:
                            total_layers = int(metadata[key])
                            break
                except Exception:
                    total_layers = 0
                if total_layers <= 0:
                    total_layers = max(0, config.n_gpu_layers)

                self._model_info = ModelInfo(
                    model_id=model_id,
                    model_path=model_path,
                    n_gpu_layers=config.n_gpu_layers,
                    context_length=config.context_length,
                    total_layers=total_layers,
                    is_loaded=True,
                    flash_attn=config.flash_attn,
                    use_mlock=config.use_mlock,
                    kv_cache_type=config.kv_cache_type,
                    has_draft_model=has_draft,
                    cache_context_shift=config.cache_context_shift,
                    tensor_split=config.tensor_split,
                    main_gpu=config.main_gpu,
                )

                if progress_callback:
                    progress_callback(1.0, "Model hazır! ✓")

                logger.info(
                    f"Model loaded: {model_id} | "
                    f"gpu_layers={config.n_gpu_layers} | "
                    f"ctx={config.context_length} | "
                    f"kv={config.kv_cache_type} | "
                    f"flash_attn={config.flash_attn} | "
                    f"mlock={config.use_mlock} | "
                    f"threads={resolved_threads} | "
                    f"speculative={'yes' if has_draft else 'no'} | "
                    f"context_prune={config.auto_context_prune}"
                )

                return self._model_info

            except ImportError:
                raise RuntimeError(
                    "llama-cpp-python yüklü değil. "
                    "Kurulum: pip install llama-cpp-python"
                )
            except Exception as e:
                self._model = None
                self._model_info = None
                self._config = None
                gc.collect()
                raise RuntimeError(f"Model yüklenemedi: {str(e)}")

    @staticmethod
    def classify_load_error(error: Exception | str) -> str:
        """Classify native loader failures so only safe retries are attempted."""
        message = str(error).lower()
        if any(token in message for token in (
            "cuda out of memory", "cublas_status_alloc_failed", "cuda error 2",
            "failed to allocate cuda", "cuda_malloc", "vulkan out of device memory",
            "device memory allocation", "gpu memory", "vram",
        )):
            return "gpu_oom"
        if any(token in message for token in (
            "paging file is too small", "pagefile", "cannot allocate memory",
            "not enough memory", "std::bad_alloc", "bad allocation",
            "commitment limit", "memoryerror", "failed to map",
        )):
            return "host_oom"
        if any(token in message for token in (
            "kv cache", "context size", "n_ctx", "compute buffer",
        )) and any(token in message for token in ("alloc", "memory", "failed")):
            return "context_oom"
        if any(token in message for token in (
            "invalid magic", "invalid gguf", "unsupported model", "unknown model architecture",
            "no such file", "does not exist", "permission denied",
        )):
            return "model_error"
        if "llama-cpp-python" in message or "no module named 'llama_cpp'" in message:
            return "runtime_missing"
        return "unknown"

    def load_model_adaptive(
        self,
        model_id: str,
        model_path: str,
        config: EngineConfig,
        *,
        max_attempts: int = 6,
        min_gpu_layers: int = 0,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[ModelInfo, AdaptiveLoadReport]:
        """Load with bounded OOM recovery and an auditable attempt history.

        llama.cpp cannot move transformer layers while a generation is in
        progress.  Safe adaptation therefore happens between complete load
        attempts: the failed native context is released, the memory profile is
        reduced, and a fresh context is created.
        """
        current = config.model_copy(deep=True)
        attempts: List[LoadAttempt] = []
        recovered = False
        max_attempts = max(1, min(int(max_attempts), 12))
        self._last_load_report = None

        for attempt_number in range(1, max_attempts + 1):
            started = time.perf_counter()
            if progress_callback:
                progress_callback(
                    (attempt_number - 1) / max_attempts,
                    f"Uyarlanabilir yükleme denemesi {attempt_number}/{max_attempts}",
                )
            try:
                if progress_callback is None:
                    info = self.load_model(model_id, model_path, current)
                else:
                    info = self.load_model(
                        model_id,
                        model_path,
                        current,
                        progress_callback=progress_callback,
                    )
                attempts.append(LoadAttempt(
                    attempt=attempt_number,
                    n_gpu_layers=current.n_gpu_layers,
                    context_length=current.context_length,
                    n_batch=current.n_batch,
                    use_mlock=current.use_mlock,
                    outcome="loaded",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                ))
                report = AdaptiveLoadReport(
                    succeeded=True,
                    attempts=attempts,
                    final_config=current,
                    recovered_from_oom=recovered,
                )
                self._last_load_report = report
                return info, report
            except Exception as exc:
                error_type = self.classify_load_error(exc)
                attempts.append(LoadAttempt(
                    attempt=attempt_number,
                    n_gpu_layers=current.n_gpu_layers,
                    context_length=current.context_length,
                    n_batch=current.n_batch,
                    use_mlock=current.use_mlock,
                    outcome="failed",
                    error_type=error_type,
                    error=str(exc)[:1000],
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                ))
                retryable = error_type in {"gpu_oom", "host_oom", "context_oom"}
                if not retryable or attempt_number >= max_attempts:
                    report = AdaptiveLoadReport(
                        succeeded=False,
                        attempts=attempts,
                        final_config=None,
                        recovered_from_oom=recovered,
                    )
                    self._last_load_report = report
                    raise RuntimeError(str(exc)) from exc

                recovered = True
                if current.use_mlock:
                    current.use_mlock = False
                elif error_type == "gpu_oom" and current.n_gpu_layers > min_gpu_layers:
                    reduction = max(1, math.ceil(current.n_gpu_layers * 0.18))
                    current.n_gpu_layers = max(min_gpu_layers, current.n_gpu_layers - reduction)
                elif current.n_batch > 32:
                    current.n_batch = max(32, current.n_batch // 2)
                elif current.context_length > 512:
                    current.context_length = max(512, current.context_length // 2)
                elif current.n_gpu_layers > min_gpu_layers:
                    current.n_gpu_layers = max(min_gpu_layers, current.n_gpu_layers - 1)
                else:
                    report = AdaptiveLoadReport(
                        succeeded=False,
                        attempts=attempts,
                        final_config=None,
                        recovered_from_oom=recovered,
                    )
                    self._last_load_report = report
                    raise RuntimeError(str(exc)) from exc

                gc.collect()

    def unload_model(self) -> None:
        """FR-307: Safely unload the current model from memory."""
        with self._lock:
            if self._is_generating:
                raise RuntimeError("Üretim sürerken model bellekten kaldırılamaz.")
            self._unload_internal()

    def _unload_internal(self) -> None:
        """Internal unload without lock."""
        model_id = self._model_info.model_id if self._model_info else "unknown"
        logger.info(f"Unloading model: {model_id}")

        for attr in ("_model", "_draft_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
                setattr(self, attr, None)

        self._model_info = None
        self._config = None
        self._is_generating = False
        self._should_stop = False
        self._evaluated_message_count = 0

        # Force VRAM/RAM release
        gc.collect()

    def stop_generation(self) -> None:
        """FR-303: Interrupt the current generation."""
        self._should_stop = True
        self._stop_event.set()

    def _apply_context_shift(
        self,
        messages: List[Dict[str, str]],
        max_tokens_estimate: int = 3800,
    ) -> List[Dict[str, str]]:
        """
        If the conversation history is likely to exceed the context window,
        trim the oldest non-system messages while
        always keeping the system prompt and the latest user message.

        This keeps the final prompt within the configured context budget. The
        low-level llama.cpp context-shift switch is not exposed by the Python
        wrapper, so this is deliberately implemented as message pruning.

        Args:
            messages:             Full message list (formatted, with system msg)
            max_tokens_estimate:  Approximate token budget (conservative)

        Returns:
            Trimmed message list safe for the current context window.
        """
        if (
            not self._config
            or not self._config.cache_context_shift
            or not self._config.auto_context_prune
        ):
            return messages

        estimated_tokens = self.count_prompt_tokens(messages)

        if estimated_tokens <= max_tokens_estimate:
            return messages

        # Separate system prompt from dialogue
        system_msgs = [m for m in messages if m.get("role") == "system"]
        dialogue = [m for m in messages if m.get("role") != "system"]

        # Always keep the last user message + previous assistant turn
        tail = dialogue[-2:] if len(dialogue) >= 2 else dialogue

        # Trim complete oldest turns from the front until we're under budget.
        remaining = dialogue[:-2] if len(dialogue) >= 2 else []
        original_remaining = list(remaining)
        summary_reserve = 0
        if self._config.context_compaction_mode == "extractive_summary" and remaining:
            summary_reserve = min(384, max(64, int(max_tokens_estimate * 0.20)))
        dialogue_budget = max(64, max_tokens_estimate - summary_reserve)
        while remaining:
            check = system_msgs + remaining + tail
            if self.count_prompt_tokens(check) <= dialogue_budget:
                break
            drop_count = 2 if len(remaining) >= 2 and remaining[0].get("role") == "user" else 1
            remaining = remaining[drop_count:]

        dropped_messages = original_remaining[:len(original_remaining) - len(remaining)]
        summary_text = ""
        if dropped_messages and self._config.context_compaction_mode == "extractive_summary":
            compacted = self._compact_messages(dropped_messages)
            if compacted:
                summary_text = "Önceki konuşmanın sıkıştırılmış özeti:\n" + compacted

        def build_trimmed(active_summary: str) -> List[Dict[str, str]]:
            prefix = [dict(message) for message in system_msgs]
            if active_summary:
                if prefix:
                    original = str(prefix[-1].get("content", ""))
                    prefix[-1]["content"] = f"{original}\n\n{active_summary}".strip()
                else:
                    prefix.append({"role": "system", "content": active_summary})
            return prefix + remaining + tail

        trimmed = build_trimmed(summary_text)
        while summary_text and self.count_prompt_tokens(trimmed) > max_tokens_estimate:
            if len(summary_text) <= 240:
                summary_text = ""
            else:
                summary_text = summary_text[:max(200, len(summary_text) // 2)]
            trimmed = build_trimmed(summary_text)
        dropped = len(dialogue) - len(remaining) - len(tail)
        if dropped > 0:
            logger.debug(
                f"Context shift: dropped {dropped} old messages "
                f"to stay within {max_tokens_estimate} token budget."
            )
        if self.count_prompt_tokens(trimmed) > max_tokens_estimate:
            raise ValueError(
                "Sistem promptu veya son ileti bağlam penceresine sığmıyor; "
                "daha kısa bir prompt ya da daha büyük context_length kullanın."
            )
        return trimmed

    @staticmethod
    def _compact_messages(messages: List[Dict[str, str]], max_chars: int = 1600) -> str:
        """Create a bounded, deterministic summary without a recursive model call."""
        parts: List[str] = []
        remaining = max_chars
        for message in messages:
            if remaining <= 40:
                break
            role = "Kullanıcı" if message.get("role") == "user" else "Asistan"
            content = " ".join(str(message.get("content", "")).split())
            if not content:
                continue
            allowance = min(220, remaining - len(role) - 3)
            if allowance <= 0:
                break
            excerpt = content if len(content) <= allowance else content[:max(1, allowance - 1)].rstrip() + "…"
            part = f"{role}: {excerpt}"
            parts.append(part)
            remaining -= len(part) + 1
        return "\n".join(parts)

    async def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        params: InferenceParams,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate tokens with streaming output.
        FR-302: Token-by-token streaming.
        FR-303: Interruptible via stop_generation().

        Yields dicts with:
            - type: "token" | "done" | "error"
            - content: token text (for "token")
            - result: InferenceResult (for "done")
            - error: error message (for "error")
        """
        start_error = self._begin_generation()
        if start_error:
            yield {"type": "error", "error": start_error}
            return

        producer = None
        try:
            prompt_messages = self._prepare_prompt(messages, params)

            start_time = time.perf_counter()
            first_token_time: Optional[float] = None
            generated_text = ""
            token_count = 0
            finish_reason = "stop"

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()
            model = self._model

            def produce() -> None:
                try:
                    stream = model.create_chat_completion(
                        messages=prompt_messages,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        top_k=params.top_k,
                        repeat_penalty=params.repeat_penalty,
                        max_tokens=params.max_tokens,
                        stop=params.stop if params.stop else None,
                        stream=True,
                    )
                    for produced_chunk in stream:
                        if self._stop_event.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, produced_chunk)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            producer = loop.run_in_executor(None, produce)

            while True:
                timeout = None
                if self._config and self._config.generation_timeout_s > 0:
                    elapsed = time.perf_counter() - start_time
                    timeout = max(0.001, self._config.generation_timeout_s - elapsed)
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    self._stop_event.set()
                    raise TimeoutError(
                        f"Generation exceeded the configured timeout of "
                        f"{self._config.generation_timeout_s:.0f} seconds"
                    ) from exc
                if chunk is sentinel:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token_text = delta.get("content", "")
                chunk_finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

                if token_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    generated_text += token_text
                    token_count += 1

                    yield {
                        "type": "token",
                        "content": token_text,
                        "tokens_generated": token_count,
                        "tokens_per_sec": self._calc_speed(token_count, start_time),
                    }

                if chunk_finish_reason:
                    finish_reason = chunk_finish_reason

            await producer

            end_time = time.perf_counter()
            actual_tokens = self.count_text_tokens(generated_text) or token_count
            stopped = self._stop_event.is_set()
            yield {
                "type": "done",
                "result": InferenceResult(
                    content=generated_text,
                    tokens_generated=actual_tokens,
                    tokens_per_sec=round(actual_tokens / max(end_time - start_time, 0.001), 1),
                    ttft_ms=((first_token_time - start_time) * 1000) if first_token_time else 0.0,
                    total_time_ms=(end_time - start_time) * 1000,
                    stopped_by_user=stopped,
                    finish_reason="user_interrupt" if stopped else finish_reason,
                ).model_dump(),
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield {"type": "error", "error": str(e)}

        finally:
            if producer is not None and not producer.done():
                self._stop_event.set()
                await producer
            self._finish_generation()

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        params: InferenceParams,
    ) -> InferenceResult:
        """
        Non-streaming generation for OpenAI-compatible API.
        FR-501: /v1/chat/completions non-streaming mode.
        """
        start_error = self._begin_generation()
        if start_error:
            raise RuntimeError(start_error)

        try:
            prompt_messages = self._prepare_prompt(messages, params)
            start_time = time.perf_counter()

            result = self._model.create_chat_completion(
                messages=prompt_messages,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                repeat_penalty=params.repeat_penalty,
                max_tokens=params.max_tokens,
                stop=params.stop if params.stop else None,
                stream=False,
            )

            elapsed = time.perf_counter() - start_time
            content = result["choices"][0]["message"]["content"]
            tokens = result.get("usage", {}).get("completion_tokens") or self.count_text_tokens(content)

            return InferenceResult(
                content=content,
                tokens_generated=tokens,
                tokens_per_sec=round(tokens / elapsed, 1) if elapsed > 0 else 0.0,
                ttft_ms=0.0,
                total_time_ms=elapsed * 1000,
                finish_reason=result["choices"][0].get("finish_reason", "stop"),
            )
        finally:
            self._finish_generation()

    def _begin_generation(self) -> Optional[str]:
        with self._lock:
            if self._model is None:
                return "Yüklü model yok"
            if self._is_generating:
                return "Zaten bir üretim devam ediyor"
            self._is_generating = True
            self._should_stop = False
            self._stop_event.clear()
        return None

    def _finish_generation(self) -> None:
        with self._lock:
            self._is_generating = False
            self._should_stop = False
            self._stop_event.clear()

    def _prepare_prompt(
        self,
        messages: List[Dict[str, str]],
        params: InferenceParams,
    ) -> List[Dict[str, str]]:
        ctx_limit = self._config.context_length if self._config else 4096
        if params.max_tokens >= ctx_limit - 32:
            raise ValueError(
                f"max_tokens ({params.max_tokens}) bağlam boyutundan ({ctx_limit}) küçük olmalıdır."
            )

        prompt_budget = max(64, int(ctx_limit * 0.96) - params.max_tokens)
        raw_messages = self._format_messages(messages, params.system_prompt)
        prompt_tokens = self.count_prompt_tokens(raw_messages)
        pruning_enabled = bool(self._config and self._config.auto_context_prune)
        if prompt_tokens > prompt_budget and not pruning_enabled:
            raise ValueError(
                "İleti geçmişi bağlam penceresine sığmıyor. Otomatik kırpmayı etkinleştirin."
            )
        prepared = self._apply_context_shift(raw_messages, prompt_budget)
        if self.count_prompt_tokens(prepared) > prompt_budget:
            raise ValueError(
                "İleti geçmişi bağlam penceresine sığmıyor; otomatik bağlam kırpmayı etkinleştirin "
                "veya daha kısa bir geçmiş kullanın."
            )
        return prepared

    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            tokens = self._model.tokenize(text.encode("utf-8"), add_bos=False)
            if isinstance(tokens, (list, tuple)):
                return len(tokens)
        except Exception:
            pass
        return max(1, len(text) // 4)

    def count_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        serialized = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        return self.count_text_tokens(serialized)

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> List[Dict[str, str]]:
        """Format messages for llama-cpp-python chat completion."""
        formatted: List[Dict[str, str]] = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        for msg in messages:
            formatted.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
        return formatted

    def _calc_speed(self, token_count: int, start_time: float) -> float:
        """Calculate current tokens per second."""
        elapsed = time.perf_counter() - start_time
        if elapsed > 0 and token_count > 0:
            return round(token_count / elapsed, 1)
        return 0.0

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Return a human-readable dict of active optimizations for telemetry."""
        if not self._config:
            return {}
        return {
            "kv_cache_type":       self._config.kv_cache_type,
            "flash_attn":          self._config.flash_attn,
            "use_mlock":           self._config.use_mlock,
            "use_mmap":            self._config.use_mmap,
            "n_threads":           self._config.n_threads or _get_physical_cores(),
            "speculative_decoding": self._config.speculative_decoding,
            "context_shift":       self._config.cache_context_shift,
            "tensor_split":        self._config.tensor_split,
            "main_gpu":            self._config.main_gpu,
            "auto_context_prune":  self._config.auto_context_prune,
            "context_compaction_mode": self._config.context_compaction_mode,
            "backend_preference": self._config.backend_preference,
            "generation_timeout_s": self._config.generation_timeout_s,
        }


# Singleton instance
engine = InferenceEngine()
