"""
Base inference backend interface for SLM-Bench.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time


class GenerationResult:
    """Detailed result of a text generation call."""
    def __init__(self, text: str, latency_ms: float, tokens_generated: int = 0, reused: bool = False):
        self.text = text
        self.latency_ms = latency_ms
        self.tokens_generated = tokens_generated
        self.reused = reused
        self.tokens_per_sec = (tokens_generated / (latency_ms / 1000.0)) if latency_ms > 0 and tokens_generated > 0 else 0.0


class BaseInferenceBackend(ABC):
    """Abstract base class for all local and hosted inference backends."""

    def __init__(self, model_id: str, quantization: str = "BF16", device: str = "auto", **kwargs):
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        self.extra_kwargs = kwargs
        self.is_loaded = False
        self.on_stream_chunk = kwargs.get("on_stream_chunk")
        self.on_generation_result = kwargs.get("on_generation_result")

    def set_stream_callback(self, callback) -> None:
        """Set a real-time token/chunk streaming callback."""
        self.on_stream_chunk = callback

    def set_generation_callback(self, callback) -> None:
        """Set a post-generation result telemetry callback."""
        self.on_generation_result = callback

    @abstractmethod
    def load(self) -> None:
        """Load the model weights into memory/VRAM according to quantization strategy."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text completion from a raw prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate completion following chat conversational message format."""
        pass

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate tool call response.
        Default implementation converts tools to a JSON schema prompt format
        and parses model response for function calling.
        """
        tool_prompt = (
            "\nYou have access to the following tools:\n"
            f"{tools}\n"
            "If you need to call a tool, respond ONLY with a JSON object in this format:\n"
            '{"name": "<function_name>", "arguments": {<args>}}\n'
            "If no tool is needed, respond with normal text.\n"
        )
        augmented_messages = list(messages)
        if augmented_messages and augmented_messages[0].get("role") == "system":
            augmented_messages[0]["content"] += tool_prompt
        else:
            augmented_messages.insert(0, {"role": "system", "content": tool_prompt})

        gen = self.chat(augmented_messages, max_new_tokens=max_new_tokens, temperature=0.0)
        return {"raw_output": gen.text, "latency_ms": gen.latency_ms}

    def get_model_ram_gb(self) -> Optional[float]:
        """Return measured RAM or VRAM footprint in GB if available from runtime."""
        return getattr(self, "model_ram_gb", None)

    def unload(self) -> None:
        """Release resources."""
        self.is_loaded = False
