from dataclasses import dataclass, field
from typing import Any, Protocol

@dataclass
class ModelResponse:
    text: str
    model: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

class ModelProvider(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse: ...

class MemoryProvider(Protocol):
    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

class ToolProvider(Protocol):
    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...

class Evaluator(Protocol):
    def evaluate(self, value: Any) -> float: ...

class ComputeBackend(Protocol):
    name: str
    def available(self) -> bool: ...
