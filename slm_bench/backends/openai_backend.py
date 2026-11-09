"""
OpenAI-compatible inference backend for SLM-Bench.
Supports local engines like Ollama, LM Studio, vLLM, LocalAI, Jan, text-generation-webui,
or any custom runner exposing an OpenAI-compatible REST API (/v1/chat/completions or /v1/completions).
"""

import json
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from .base import BaseInferenceBackend, GenerationResult
from ..utils.logger import logger


class OpenAICompatibleBackend(BaseInferenceBackend):
    """
    Inference backend communicating with an OpenAI-compatible REST API.
    Ideal for testing local models hosted via Ollama, LM Studio, vLLM,
    or custom software runtime environments.
    """

    def __init__(
        self,
        model_id: str,
        quantization: str = "Custom / Server",
        device: str = "auto",
        endpoint_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        reasoning_effort: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        thinking: str = "auto",
        truncation_policy: str = "retry",
        max_token_budget: int = 16384,
        max_retries: int = 10,
        budget_manager=None,
        **kwargs,
    ):
        super().__init__(model_id=model_id, quantization=quantization, device=device, **kwargs)
        # Normalize endpoint URL (strip trailing slashes)
        base = endpoint_url or kwargs.get("base_url") or "http://localhost:11434/v1"
        self.endpoint_url = base.rstrip("/")
        self.api_key = api_key or kwargs.get("token") or "EMPTY"
        self.timeout = float(timeout)
        self.reasoning_effort = reasoning_effort or kwargs.get("reasoning_effort")
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens else None
        self.thinking = (thinking or "auto").lower()
        self.truncation_policy = (truncation_policy or "retry").lower()
        self.max_token_budget = int(max_token_budget or 4096)
        self.max_retries = max(0, int(max_retries))
        self.budget_manager = budget_manager
        self.current_task_id: Optional[str] = None
        if self.thinking not in ("auto", "on", "off"):
            raise ValueError("thinking must be one of: auto, on, off")
        if self.truncation_policy not in ("retry", "invalid", "accept"):
            raise ValueError("truncation_policy must be one of: retry, invalid, accept")
        self.model_ram_gb: Optional[float] = kwargs.get("model_ram_gb")
        self.server_engine: Optional[str] = None
        self.server_version: Optional[str] = None
        self.server_runtime: Optional[str] = None

    def load(self) -> None:
        """Verify endpoint connectivity, model availability, and runtime environment if supported."""
        logger.info(f"Connecting to OpenAI-compatible endpoint at: [cyan]{self.endpoint_url}[/cyan] for model '[bold green]{self.model_id}[/bold green]'...")
        # Optional ping to /models
        try:
            req = urllib.request.Request(
                f"{self.endpoint_url}/models",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
                    for m in data.get("data", []):
                        if isinstance(m, dict) and m.get("owned_by"):
                            self.server_engine = m.get("owned_by")
                            break
                    logger.info(f"Connected to endpoint successfully. Found {len(models)} model(s).")
        except Exception as e:
            logger.debug(f"Endpoint /models check note (endpoint might only serve completions directly): {e}")

        # Optional probe to /version (e.g. vLLM /version endpoint)
        try:
            base_url = self.endpoint_url.rsplit("/v1", 1)[0]
            req_ver = urllib.request.Request(
                f"{base_url}/version",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req_ver, timeout=2) as resp:
                if resp.status == 200:
                    v_data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(v_data, dict) and "version" in v_data:
                        self.server_version = v_data["version"]
        except Exception:
            pass

        # Synthesize server runtime descriptor
        from ..models.hardware import is_endpoint_wsl_relayed, is_wsl_environment
        engine_str = self.server_engine or "OpenAI API"
        if engine_str.lower() == "vllm":
            engine_str = "vLLM"
        if self.server_version:
            engine_str = f"{engine_str} {self.server_version}"
        if is_endpoint_wsl_relayed(self.endpoint_url) or is_wsl_environment():
            engine_str = f"{engine_str} in WSL2"
        self.server_runtime = engine_str

        self.is_loaded = True
        logger.info(f"OpenAI-compatible backend loaded ({self.server_runtime}) and ready for '{self.model_id}'.")

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "slm-bench/0.1.0",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON POST request to the OpenAI-compatible endpoint."""
        url = f"{self.endpoint_url}/{path.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if e.fp else str(e)
            raise RuntimeError(f"HTTP error {e.code} from {url}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI-compatible endpoint at {url}: {e}") from e

    def _chat_payload(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": max(0.0, temperature),
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if stop_sequences:
            payload["stop"] = stop_sequences
        if self.thinking in ("on", "off"):
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking == "on"}
        return payload

    def _parse_chat_response(self, resp_data: Dict[str, Any], latency_ms: float) -> GenerationResult:
        choices = resp_data.get("choices", [])
        if not choices:
            return GenerationResult("", latency_ms, 0)
        first_choice = choices[0]
        message = first_choice.get("message", {})
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        finish_reason = first_choice.get("finish_reason")
        usage = resp_data.get("usage", {}) or {}
        tokens = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens") or 0
        if not tokens:
            tokens = len(text.split()) + len(reasoning.split())
        return GenerationResult(
            text,
            latency_ms,
            tokens,
            reasoning_text=reasoning,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
        )

    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]],
    ):
        payload = self._chat_payload(messages, max_new_tokens, temperature, stop_sequences)
        start_time = time.perf_counter()
        try:
            resp_data = self._post("chat/completions", payload)
        except Exception:
            prompt_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
            payload_comp = {
                "model": self.model_id,
                "prompt": prompt_text,
                "max_tokens": max_new_tokens,
                "temperature": max(0.0, temperature),
            }
            if stop_sequences:
                payload_comp["stop"] = stop_sequences
            resp_data = self._post("completions", payload_comp)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            choices = resp_data.get("choices", [])
            text = choices[0].get("text", "") if choices else ""
            finish_reason = choices[0].get("finish_reason") if choices else None
            tokens = resp_data.get("usage", {}).get("completion_tokens", len(text.split()))
            return (
                GenerationResult(
                    text,
                    latency_ms,
                    tokens,
                    finish_reason=finish_reason,
                    truncated=finish_reason == "length",
                ),
                True,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_chat_response(resp_data, latency_ms), False

    def _stream_once(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]],
    ) -> Optional[GenerationResult]:
        payload = self._chat_payload(messages, max_new_tokens, temperature, stop_sequences)
        payload["stream"] = True
        url = f"{self.endpoint_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="POST")
        start_time = time.perf_counter()
        collected_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        finish_reason = None
        usage_tokens = None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str == "data: [DONE]":
                        break
                    if not line_str.startswith("data: "):
                        continue
                    try:
                        chunk_data = json.loads(line_str[6:])
                    except Exception:
                        continue
                    choices = chunk_data.get("choices", [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            collected_chunks.append(content)
                            self.on_stream_chunk(content)
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if reasoning:
                            reasoning_chunks.append(reasoning)
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
                    usage = chunk_data.get("usage") or {}
                    if usage.get("completion_tokens"):
                        usage_tokens = usage.get("completion_tokens")
        except Exception as exc:
            logger.debug(f"Streaming probe note, falling back to synchronous execution: {exc}")
            return None

        full_text = "".join(collected_chunks)
        reasoning_text = "".join(reasoning_chunks)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        tokens = usage_tokens if usage_tokens is not None else len(full_text.split()) + len(reasoning_text.split())
        return GenerationResult(
            full_text,
            latency_ms,
            tokens,
            reasoning_text=reasoning_text,
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
        )

    def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        stop_sequences: Optional[List[str]],
    ) -> Optional[GenerationResult]:
        total_latency_ms = 0.0
        total_tokens = 0
        total_reasoning_tokens = 0
        budget = max_new_tokens
        last: Optional[GenerationResult] = None
        for attempt in range(self.max_retries + 1):
            last = self._stream_once(messages, budget, temperature, stop_sequences)
            if last is None:
                return None
            total_latency_ms += last.latency_ms
            total_tokens += last.tokens_generated
            total_reasoning_tokens += last.reasoning_tokens
            should_retry = (
                self.truncation_policy == "retry"
                and last.truncated
                and budget < self.max_token_budget
                and attempt < self.max_retries
            )
            if not should_retry:
                break
            next_budget = min(budget * 2, self.max_token_budget)
            if next_budget <= budget:
                break
            budget = next_budget

        final = GenerationResult(
            last.text,
            total_latency_ms,
            total_tokens,
            reasoning_text=last.reasoning_text,
            reasoning_tokens=total_reasoning_tokens,
            finish_reason=last.finish_reason,
            truncated=last.truncated,
            final_tokens=last.tokens_generated,
            budget_start=max_new_tokens,
            final_budget=budget,
            cap_exceeded=last.truncated and budget >= self.max_token_budget,
        )
        if getattr(self, "on_generation_result", None):
            self.on_generation_result(
                {"text": final.text, "latency_ms": final.latency_ms, "tokens": final.tokens_generated}
            )
        return final

    def _record_budget(self, base_budget: int, start_budget: int, last: Optional[GenerationResult]) -> None:
        if self.budget_manager is None or not self.current_task_id or last is None:
            return
        self.budget_manager.record(
            self.current_task_id,
            base_budget,
            start_budget,
            self.max_token_budget,
            last.final_tokens,
            last.truncated,
            last.final_budget,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        base_budget = max_new_tokens
        if self.max_new_tokens:
            base_budget = max(base_budget, self.max_new_tokens)
        start_budget = base_budget
        if self.budget_manager is not None and self.current_task_id:
            start_budget = self.budget_manager.start_for(
                self.current_task_id, base_budget, self.max_token_budget
            )

        if getattr(self, "on_stream_chunk", None):
            streamed = self._stream_chat(messages, start_budget, temperature, stop_sequences)
            if streamed is not None:
                self._record_budget(base_budget, start_budget, streamed)
                return streamed

        total_latency_ms = 0.0
        total_tokens = 0
        total_reasoning_tokens = 0
        budget = start_budget
        last: Optional[GenerationResult] = None

        for attempt in range(self.max_retries + 1):
            last, used_fallback = self._chat_once(messages, budget, temperature, stop_sequences)
            total_latency_ms += last.latency_ms
            total_tokens += last.tokens_generated
            total_reasoning_tokens += last.reasoning_tokens
            if used_fallback:
                break
            should_retry = (
                self.truncation_policy == "retry"
                and last.truncated
                and budget < self.max_token_budget
                and attempt < self.max_retries
            )
            if not should_retry:
                break
            next_budget = min(budget * 2, self.max_token_budget)
            if next_budget <= budget:
                break
            budget = next_budget

        final = GenerationResult(
            last.text,
            total_latency_ms,
            total_tokens,
            reasoning_text=last.reasoning_text,
            reasoning_tokens=total_reasoning_tokens,
            finish_reason=last.finish_reason,
            truncated=last.truncated,
            final_tokens=last.tokens_generated,
            budget_start=start_budget,
            final_budget=budget,
            cap_exceeded=last.truncated and budget >= self.max_token_budget,
        )
        self._record_budget(base_budget, start_budget, final)
        if getattr(self, "on_generation_result", None):
            self.on_generation_result(
                {"text": final.text, "latency_ms": final.latency_ms, "tokens": final.tokens_generated}
            )
        return final

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text from a raw prompt using the chat endpoint."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )

    def get_model_ram_gb(self) -> Optional[float]:
        """Automatically retrieve model RAM/VRAM footprint from Ollama or local server."""
        if self.model_ram_gb:
            return self.model_ram_gb
        from ..models.memory_detector import detect_ollama_model_ram_gb, detect_runner_process_ram_gb
        ram = detect_ollama_model_ram_gb(self.endpoint_url, self.model_id)
        if ram:
            return ram
        return detect_runner_process_ram_gb()

    def unload(self) -> None:
        """Release backend resources."""
        self.is_loaded = False
