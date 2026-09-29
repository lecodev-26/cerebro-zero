from dataclasses import dataclass, field

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
