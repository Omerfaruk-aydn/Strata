"""
AI Runner — Inference Engine
Wraps llama-cpp-python for model loading and token generation.
Implements FR-301 through FR-307.
"""

import asyncio
import time
import threading
from typing import Optional, AsyncGenerator, Dict, Any, Callable, List
from pydantic import BaseModel, field_validator
import logging

logger = logging.getLogger(__name__)


class InferenceParams(BaseModel):
    """Generation parameters (FR-306)."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    stop: List[str] = []
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
    ttft_ms: float = 0.0   # Time to first token
    total_time_ms: float = 0.0
    stopped_by_user: bool = False
    finish_reason: str = "stop"  # "stop", "length", "user_interrupt"


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


class InferenceEngine:
    """
    Manages LLM inference using llama-cpp-python.
    Enforces single active model (FR-307).
    Supports streaming with interrupt (FR-302, FR-303).
    """

    def __init__(self):
        self._model = None
        self._model_info: Optional[ModelInfo] = None
        self._is_generating = False
        self._should_stop = False
        self._stop_event = threading.Event()  # For test introspection
        self._lock = threading.Lock()
        self._load_progress_callback: Optional[Callable] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @property
    def model_info(self) -> Optional[ModelInfo]:
        return self._model_info

    def load_model(
        self,
        model_id: str,
        model_path: str,
        n_gpu_layers: int = -1,
        context_length: int = 4096,
        n_threads: Optional[int] = None,
        use_mmap: bool = True,
        n_batch: int = 512,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> ModelInfo:
        """
        Load a GGUF model into memory.
        FR-301: n_gpu_layers auto or manual override.
        FR-307: Unloads previous model first.

        Args:
            model_id: Model identifier
            model_path: Path to .gguf file
            n_gpu_layers: -1 = auto (all to GPU), 0 = CPU only, N = specific
            context_length: Context window size
            n_threads: CPU threads for inference (None = auto)
            use_mmap: Whether to use memory-mapped I/O
            n_batch: Batch size for prompt processing
            progress_callback: Called with (progress_pct, status_message)
        """
        with self._lock:
            # FR-307: Unload previous model
            if self._model is not None:
                self._unload_internal()

            if progress_callback:
                progress_callback(0.0, "Model yükleniyor...")

            try:
                from llama_cpp import Llama

                if progress_callback:
                    progress_callback(0.1, "llama.cpp başlatılıyor...")

                # Build kwargs
                kwargs = {
                    "model_path": model_path,
                    "n_gpu_layers": n_gpu_layers,
                    "n_ctx": context_length,
                    "use_mmap": use_mmap,
                    "n_batch": n_batch,
                    "verbose": False,
                }

                if n_threads is not None:
                    kwargs["n_threads"] = n_threads

                if progress_callback:
                    progress_callback(0.3, "Model belleğe yükleniyor...")

                self._model = Llama(**kwargs)

                if progress_callback:
                    progress_callback(0.9, "Model hazırlanıyor...")

                # Extract actual model metadata
                total_layers = n_gpu_layers if n_gpu_layers >= 0 else 0
                try:
                    metadata = self._model.metadata
                    if "llama.block_count" in metadata:
                        total_layers = int(metadata["llama.block_count"])
                except Exception:
                    pass

                self._model_info = ModelInfo(
                    model_id=model_id,
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    context_length=context_length,
                    total_layers=total_layers,
                    is_loaded=True,
                )

                if progress_callback:
                    progress_callback(1.0, "Model hazır!")

                logger.info(
                    f"Model loaded: {model_id}, "
                    f"gpu_layers={n_gpu_layers}, ctx={context_length}"
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
                raise RuntimeError(f"Model yüklenemedi: {str(e)}")

    def unload_model(self) -> None:
        """FR-307: Safely unload the current model from memory."""
        with self._lock:
            self._unload_internal()

    def _unload_internal(self) -> None:
        """Internal unload without lock."""
        if self._model is not None:
            logger.info(f"Unloading model: {self._model_info.model_id if self._model_info else 'unknown'}")
            try:
                del self._model
            except Exception:
                pass
            self._model = None
            self._model_info = None
            self._is_generating = False
            self._should_stop = False

            # Force garbage collection to free VRAM
            import gc
            gc.collect()

    def stop_generation(self) -> None:
        """FR-303: Interrupt the current generation."""
        self._should_stop = True
        self._stop_event.set()

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
        if self._model is None:
            yield {"type": "error", "error": "Yüklü model yok"}
            return

        if self._is_generating:
            yield {"type": "error", "error": "Zaten bir üretim devam ediyor"}
            return

        self._is_generating = True
        self._should_stop = False

        try:
            # Build the prompt from messages
            prompt_messages = self._format_messages(messages, params.system_prompt)

            start_time = time.perf_counter()
            first_token_time = None
            generated_text = ""
            token_count = 0

            # Run inference in a thread pool to not block the event loop
            loop = asyncio.get_event_loop()

            # Create the completion stream
            stream = await loop.run_in_executor(
                None,
                lambda: self._model.create_chat_completion(
                    messages=prompt_messages,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    repeat_penalty=params.repeat_penalty,
                    max_tokens=params.max_tokens,
                    stop=params.stop if params.stop else None,
                    stream=True,
                )
            )

            for chunk in stream:
                if self._should_stop:
                    yield {
                        "type": "done",
                        "result": InferenceResult(
                            content=generated_text,
                            tokens_generated=token_count,
                            tokens_per_sec=self._calc_speed(token_count, start_time),
                            ttft_ms=((first_token_time - start_time) * 1000) if first_token_time else 0,
                            total_time_ms=(time.perf_counter() - start_time) * 1000,
                            stopped_by_user=True,
                            finish_reason="user_interrupt",
                        ).model_dump(),
                    }
                    return

                # Extract token from chunk
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token_text = delta.get("content", "")
                finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

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

                if finish_reason:
                    break

                # Small yield to allow event loop to process
                await asyncio.sleep(0)

            # Generation complete
            end_time = time.perf_counter()
            yield {
                "type": "done",
                "result": InferenceResult(
                    content=generated_text,
                    tokens_generated=token_count,
                    tokens_per_sec=self._calc_speed(token_count, start_time),
                    ttft_ms=((first_token_time - start_time) * 1000) if first_token_time else 0,
                    total_time_ms=(end_time - start_time) * 1000,
                    finish_reason="stop",
                ).model_dump(),
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield {"type": "error", "error": str(e)}

        finally:
            self._is_generating = False
            self._should_stop = False

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        params: InferenceParams,
    ) -> InferenceResult:
        """
        Non-streaming generation for OpenAI-compatible API.
        FR-501: /v1/chat/completions non-streaming mode.
        """
        if self._model is None:
            raise RuntimeError("Yüklü model yok")

        prompt_messages = self._format_messages(messages, params.system_prompt)

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

        end_time = time.perf_counter()

        content = result["choices"][0]["message"]["content"]
        tokens = result.get("usage", {}).get("completion_tokens", len(content.split()))

        return InferenceResult(
            content=content,
            tokens_generated=tokens,
            tokens_per_sec=tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            ttft_ms=0,  # Not measurable in non-streaming
            total_time_ms=(end_time - start_time) * 1000,
            finish_reason=result["choices"][0].get("finish_reason", "stop"),
        )

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> List[Dict[str, str]]:
        """Format messages for llama-cpp-python chat completion."""
        formatted = []

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


# Singleton instance
engine = InferenceEngine()
