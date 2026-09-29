from .registry import ModelRegistry
from .spec import ModelSpec
from .lineage import ModelLineage

class ModelCatalog:
    def __init__(self):
        self.registry = ModelRegistry()
        self.lineages: dict[str, ModelLineage] = {}

    def add(self, model, spec, lineage=None):
        self.registry.register(model, spec)
        if lineage: self.lineages[spec.name] = lineage

    def get(self, name): return self.registry.get(name)
    def specs(self): return self.registry.list()
    def list(self):
        return [{"name": s.name, "version": s.version, "capabilities": list(s.capabilities)} for s in self.registry.list()]
    def lineage(self, name): return self.lineages.get(name)
