import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

from .base import LocalEchoProvider, BaseModelProvider
from ..core.contracts import ModelResponse

@dataclass
class OpenAICompatibleProvider(BaseModelProvider):
    """Provider for OpenAI-compatible chat-completions APIs.

    Credentials are read from the environment and never persisted by Cerebro.
    """
    name: str = "openai-compatible"
    api_key_env: str = "CEREBRO_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-oss-120b"
    timeout: float = 60.0

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        key = os.getenv(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload = {"model": kwargs.pop("model", self.model), "messages": messages}
        for k in ("temperature", "max_tokens", "top_p", "seed"):
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "cerebro-zero/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Teacher API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Teacher API connection failed: {exc.reason}") from exc
        choices = body.get("choices") or []
        if not choices or not choices[0].get("message", {}).get("content"):
            raise RuntimeError("Teacher API returned no assistant content")
        text = choices[0]["message"]["content"]
        return ModelResponse(text=text, model=body.get("model", payload["model"]), metadata={"backend": "openai-compatible", "usage": body.get("usage", {})})

@dataclass
class OpenRouterProvider(OpenAICompatibleProvider):
    name: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"

__all__=["LocalEchoProvider","BaseModelProvider","OpenAICompatibleProvider","OpenRouterProvider"]
