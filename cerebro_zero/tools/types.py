from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(frozen=True)
class ToolSpec:
    name: str
    function: Callable[..., Any]
    description: str = ""
    permissions: frozenset[str] = frozenset()
    risk: float = 0.0
    cost: float = 1.0
    confirmation: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: Any = None
    error: str | None = None
    duration: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __eq__(self, other):
        if isinstance(other, ToolResult):
            return (self.ok, self.tool, self.output, self.error) == (other.ok, other.tool, other.output, other.error)
        return self.ok and self.output == other
