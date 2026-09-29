from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    family: str = "cerebro"
    architecture: str = "transformer"
    vocab_size: int = 0
    context_length: int = 0
    parameters: int = 0
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
