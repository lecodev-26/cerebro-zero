from dataclasses import dataclass
from typing import Any
from ..core.contracts import ModelResponse

class BaseModelProvider:
    name = "base"
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

@dataclass
class LocalEchoProvider(BaseModelProvider):
    name: str = "cerebro-local"
    def generate(self, messages, **kwargs):
        text = messages[-1]["content"] if messages else ""
        return ModelResponse(text=text, model=self.name, metadata={"backend": "local"})
