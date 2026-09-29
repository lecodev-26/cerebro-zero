from dataclasses import dataclass, field
from typing import Any
import time

from ..models.base import BaseModelProvider

@dataclass
class TeacherResult:
    prompt: str
    response: str
    teacher: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TeacherSpec:
    name: str
    provider: str
    capabilities: tuple[str, ...] = ()
    license: str = "unknown"
    terms: str = ""
    metadata: dict = field(default_factory=dict)

class TeacherRegistry:
    def __init__(self): self._teachers = {}
    def register(self, spec): self._teachers[spec.name] = spec
    def get(self, name): return self._teachers[name]
    def list(self): return list(self._teachers.values())

    @classmethod
    def default(cls):
        r = cls()
        r.register(TeacherSpec(
            name="openrouter", provider="openrouter",
            capabilities=("instruction", "reasoning", "code", "analysis"),
            license="provider-output", terms="Must be verified against the selected provider/model terms.",
        ))
        return r


class TeacherService:
    """Controlled bridge from registered teachers to dataset generation."""
    def __init__(self, registry: TeacherRegistry, providers: dict[str, BaseModelProvider] | None = None, max_tokens: int = 512, temperature: float = 0.2, retries: int = 2):
        self.registry = registry
        self.providers = providers or {}
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = max(0, int(retries))

    def generate(self, prompt: str, teacher_name: str = "openrouter", **kwargs: Any) -> TeacherResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        spec = self.registry.get(teacher_name)
        provider = self.providers.get(spec.provider)
        if provider is None:
            raise RuntimeError(f"No provider configured for teacher: {teacher_name}")
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("max_tokens", self.max_tokens)
        request_kwargs.setdefault("temperature", self.temperature)
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = provider.generate([{"role": "user", "content": prompt.strip()}], **request_kwargs)
                if not response.text.strip():
                    raise RuntimeError("Teacher returned empty content")
                metadata = dict(response.metadata)
                metadata["attempt"] = attempt + 1
                break
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                retryable = any(code in message for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "empty content"))
                if not retryable or attempt >= self.retries:
                    raise
                time.sleep(min(2.0 ** attempt, 4.0))
        else:
            raise last_error or RuntimeError("Teacher generation failed")
        metadata = dict(response.metadata)
        metadata.update({"teacher": spec.name, "provider": spec.provider, "license": spec.license})
        return TeacherResult(prompt=prompt.strip(), response=response.text.strip(), teacher=spec.name, model=response.model, metadata=metadata)

    def generate_batch(self, prompts: list[str], teacher_name: str = "openrouter", **kwargs: Any) -> list[TeacherResult]:
        return [self.generate(prompt, teacher_name=teacher_name, **kwargs) for prompt in prompts]
