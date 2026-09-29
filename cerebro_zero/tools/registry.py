from ..compat import ToolRegistry, ToolSpec, PermissionPolicy

class ToolService:
    """Stable tool boundary; execution remains policy-gated."""
    def __init__(self): self._registry = ToolRegistry(PermissionPolicy())
    def register(self, name, function, description="", permissions=None, risk=0.0, cost=1.0):
        self._registry.register(ToolSpec(name, function, description, permissions or {"safe"}, risk, cost))
    def execute(self, name, arguments=None): return self._registry.execute(name, arguments)
    def names(self): return sorted(self._registry.tools)
