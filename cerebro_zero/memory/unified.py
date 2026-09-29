from dataclasses import dataclass, field
from time import time
import re

@dataclass
class MemoryItem:
    text: str
    kind: str = "semantic"
    metadata: dict = field(default_factory=dict)
    importance: float = .5
    timestamp: float = field(default_factory=time)

class UnifiedMemory:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.working: dict[str, MemoryItem] = {}
        self.items: list[MemoryItem] = []

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"[\\wáéíóúüñ]+", str(text).lower()))

    def add(self, text, kind="semantic", metadata=None, importance=.5):
        item = MemoryItem(str(text), kind, metadata or {}, float(importance))
        self.items.append(item); self.items = self.items[-self.capacity:]
        return item

    def remember_working(self, key, value, importance=1.0):
        self.working[key] = MemoryItem(str(value), "working", {"key": key}, importance)

    def retrieve(self, query, limit=5):
        q = self._tokens(query); scored = []
        for item in self.items:
            t = self._tokens(item.text); overlap = len(q & t) / max(1, len(q | t))
            score = .75 * overlap + .25 * item.importance
            if score > 0: scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": i.text, "kind": i.kind, "metadata": i.metadata, "score": s} for s, i in scored[:limit]]

    def stats(self):
        return {"working": len(self.working), "items": len(self.items), "semantic": sum(i.kind == "semantic" for i in self.items), "episodic": sum(i.kind == "episodic" for i in self.items), "procedural": sum(i.kind == "procedural" for i in self.items)}
