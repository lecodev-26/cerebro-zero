from dataclasses import dataclass, field
from time import time

@dataclass(frozen=True)
class ToolAuditEvent:
    tool: str
    action: str
    allowed: bool
    timestamp: float = field(default_factory=time)
    reason: str = ""
    metadata: dict = field(default_factory=dict)

class ToolAuditLog:
    def __init__(self, limit=2000): self.events=[]; self.limit=limit
    def record(self, event): self.events.append(event); self.events=self.events[-self.limit:]
    def recent(self, limit=50): return list(self.events[-limit:])
    def stats(self): return {"total":len(self.events),"allowed":sum(e.allowed for e in self.events),"denied":sum(not e.allowed for e in self.events)}
