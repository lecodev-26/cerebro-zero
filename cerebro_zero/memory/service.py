from .unified import UnifiedMemory

class MemoryService:
    def __init__(self, capacity=5000):
        self.store = UnifiedMemory(capacity)

    def add(self, text, metadata=None, importance=.5, kind="semantic"):
        return self.store.add(text, kind, metadata, importance)

    def retrieve(self, query, limit=5):
        return self.store.retrieve(query, limit)

    def remember_episode(self, cycle, outcome):
        return self.add(cycle.goal, {"cycle": cycle.id, "outcome": outcome}, .7, "episodic")

    def add_procedure(self, name, steps, success):
        return self.add(name, {"steps": list(steps), "success": bool(success)}, .7, "procedural")

    def stats(self):
        return self.store.stats()

    def consolidate(self):
        return 0
