"""
Unified Inference Gateway.
Standardizes model execution across backends (llama.cpp, vLLM, transformers, OpenAI/Ollama)
and integrates real-time telemetry profiling (TTFT, decode TPS, memory).
"""

import time
from typing import List, Dict, Any, Optional, Union
from ..backends.base import BaseInferenceBackend, GenerationResult
from ..metrics.telemetry import PerformanceCollector, TelemetrySnapshot
from .checkpoint import CheckpointStore


class InferenceGateway:
    """
    Central orchestration gateway for model execution.
    Decouples benchmark evaluation tasks from specific backend implementations
    while capturing standardized performance telemetry.
    """

    def __init__(
        self,
        backend: BaseInferenceBackend,
        collector: Optional[PerformanceCollector] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
    ):
        self.backend = backend
        self.collector = collector or PerformanceCollector()
        self.capabilities = self._detect_capabilities()
        self.checkpoint_store = checkpoint_store
        self.task_id: Optional[str] = None
        self.dataset_revision = "official"
        self.reused_calls = 0

    def set_task_context(self, task_id: str, dataset_revision: str = "official") -> None:
        """Set the current benchmark provenance used by replay keying."""
        self.task_id = task_id
        self.dataset_revision = dataset_revision

    def _checkpoint_key(self, mode: str, payload: Dict[str, Any]) -> Optional[str]:
        if self.checkpoint_store is None or self.task_id is None:
            return None
        return self.checkpoint_store.make_key(
            model_id=self.checkpoint_store.model_id,
            quantization=self.checkpoint_store.quantization,
            backend=self.checkpoint_store.backend,
            backend_version=self.checkpoint_store.backend_version,
            reasoning_effort=self.checkpoint_store.reasoning_effort,
            server=self.checkpoint_store.server,
            evaluator_version=self.checkpoint_store.evaluator_version,
            task_id=self.task_id,
            dataset_revision=self.dataset_revision,
            mode=mode,
            payload=payload,
        )

    def _write_checkpoint(
        self,
        key: Optional[str],
        mode: str,
        payload: Dict[str, Any],
        result: GenerationResult,
    ) -> None:
        if key is None:
            return
        self.checkpoint_store.put({
            "key": key,
            "model_id": self.checkpoint_store.model_id,
            "quantization": self.checkpoint_store.quantization,
            "backend": self.checkpoint_store.backend,
            "backend_version": self.checkpoint_store.backend_version,
            "reasoning_effort": self.checkpoint_store.reasoning_effort,
            "server": self.checkpoint_store.server,
            "evaluator_version": self.checkpoint_store.evaluator_version,
            "task_id": self.task_id,
            "dataset_revision": self.dataset_revision,
            "mode": mode,
            "payload": payload,
            "response": result.text,
            "tokens_generated": result.tokens_generated,
            "latency_ms": result.latency_ms,
        })

    def _detect_capabilities(self) -> Dict[str, bool]:
        """Inspect backend capabilities (tools, logprobs, streaming)."""
        backend_name = getattr(self.backend, "__class__", type(self.backend)).__name__.lower()
        return {
            "supports_chat": hasattr(self.backend, "chat"),
            "supports_generate": hasattr(self.backend, "generate"),
            "supports_native_tools": hasattr(self.backend, "chat_with_tools") or "openai" in backend_name or "vllm" in backend_name,
            "supports_logprobs": "transformers" in backend_name or "vllm" in backend_name or "llamacpp" in backend_name,
            "supports_streaming": hasattr(self.backend, "set_stream_callback"),
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Execute chat completion with automatic telemetry tracking."""
        payload = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }
        key = self._checkpoint_key("chat", payload)
        if key is not None:
            cached = self.checkpoint_store.get(key)
            if cached is not None:
                self.checkpoint_store.hits += 1
                self.reused_calls += 1
                return GenerationResult(
                    cached.get("response", ""),
                    latency_ms=0.0,
                    tokens_generated=0,
                    reused=True,
                )

        start_time = time.perf_counter()
        first_token_time = None
        token_times: List[float] = []

        original_callback = getattr(self.backend, "on_stream_chunk", None)

        def stream_hook(chunk: str):
            nonlocal first_token_time
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_times.append(now)
            # Delegate to original callback if set and not self
            if original_callback and original_callback != stream_hook:
                original_callback(chunk)

        # Hook streaming if available
        if self.capabilities["supports_streaming"]:
            self.backend.set_stream_callback(stream_hook)

        try:
            res = self.backend.chat(
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )
        finally:
            if self.capabilities["supports_streaming"]:
                self.backend.set_stream_callback(original_callback)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else None

        inter_latencies = []
        if len(token_times) > 1:
            for i in range(1, len(token_times)):
                inter_latencies.append((token_times[i] - token_times[i-1]) * 1000.0)

        decode_tps = res.tokens_per_sec
        tpot_ms = (1000.0 / decode_tps) if decode_tps > 0 else (duration_ms / max(1, res.tokens_generated))

        snapshot = TelemetrySnapshot(
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=decode_tps,
            tpot_ms=tpot_ms,
            inter_token_latencies_ms=inter_latencies,
            peak_ram_mb=self.collector.get_current_ram_mb(),
            peak_vram_mb=self.collector.get_current_vram_mb(),
            tokens_prompt=sum(len(m.get("content", "").split()) for m in messages),
            tokens_generated=res.tokens_generated,
            total_duration_ms=duration_ms,
        )
        self.collector.record_snapshot(snapshot)
        self._write_checkpoint(key, "chat", payload, res)

        return res

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Execute raw text completion with telemetry tracking."""
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }
        key = self._checkpoint_key("generate", payload)
        if key is not None:
            cached = self.checkpoint_store.get(key)
            if cached is not None:
                self.checkpoint_store.hits += 1
                self.reused_calls += 1
                return GenerationResult(
                    cached.get("response", ""),
                    latency_ms=0.0,
                    tokens_generated=0,
                    reused=True,
                )

        start_time = time.perf_counter()
        res = self.backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        decode_tps = res.tokens_per_sec
        tpot_ms = (1000.0 / decode_tps) if decode_tps > 0 else (duration_ms / max(1, res.tokens_generated))

        snapshot = TelemetrySnapshot(
            decode_tokens_per_sec=decode_tps,
            tpot_ms=tpot_ms,
            peak_ram_mb=self.collector.get_current_ram_mb(),
            peak_vram_mb=self.collector.get_current_vram_mb(),
            tokens_prompt=len(prompt.split()),
            tokens_generated=res.tokens_generated,
            total_duration_ms=duration_ms,
        )
        self.collector.record_snapshot(snapshot)
        self._write_checkpoint(key, "generate", payload, res)
        return res

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute tool-calling turn with native or prompted schema format.
        """
        # If backend implements generate_with_tools directly:
        if hasattr(self.backend, "generate_with_tools"):
            return self.backend.generate_with_tools(messages, tools, max_new_tokens=max_new_tokens)
        
        # Fallback to chat with prompted tool specs
        import json
        tool_prompt = (
            "\nYou are equipped with the following tool definitions:\n"
            f"{json.dumps(tools, indent=2)}\n"
            "When invoking a tool, reply ONLY with a JSON object in this format:\n"
            '{"name": "<tool_name>", "arguments": {<args>}}\n'
            "If no tool is needed, respond directly in natural language.\n"
        )
        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {"role": "system", "content": augmented[0]["content"] + "\n" + tool_prompt}
        else:
            augmented.insert(0, {"role": "system", "content": tool_prompt})
            
        gen = self.chat(augmented, max_new_tokens=max_new_tokens, temperature=temperature)
        return {"raw_output": gen.text, "latency_ms": gen.latency_ms, "tokens_generated": gen.tokens_generated}
