from v5.runtime import MemorySystem

class MemoryService:
    """Stable 1.0 facade over the proven V5 memory primitives."""
    def __init__(self, capacity=16): self._memory = MemorySystem(capacity)
    def add(self, text, metadata=None, importance=.5):
        self._memory.add_semantic(text, metadata, importance)
    def retrieve(self, query, limit=5): return self._memory.retrieve(query, limit)
    def remember_episode(self, cycle, outcome): self._memory.remember_episode(cycle, outcome)
    def consolidate(self): return self._memory.consolidate()
    def stats(self): return self._memory.stats()
