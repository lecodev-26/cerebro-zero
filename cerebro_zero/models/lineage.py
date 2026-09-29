from dataclasses import dataclass, field
from time import time

@dataclass(frozen=True)
class ModelLineage:
    model: str
    version: str
    parent: str | None = None
    teachers: tuple[str, ...] = ()
    training_steps: int = 0
    created_at: float = field(default_factory=time)

class TeacherRegistry:
    def __init__(self): self._teachers = {}
    def register(self, name, provider, roles=()): self._teachers[name] = {"provider": provider, "roles": tuple(roles)}
    def get(self, name): return self._teachers[name]
    def list(self): return list(self._teachers)
