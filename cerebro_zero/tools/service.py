from time import monotonic
from .types import ToolSpec, ToolResult
from .audit import ToolAuditLog, ToolAuditEvent

class ToolService:
    def __init__(self, policy=None, audit=None):
        self.tools={}; self.policy=policy; self.audit=audit or ToolAuditLog()
    def register(self, spec, function=None, **kwargs):
        if isinstance(spec, str):
            spec = ToolSpec(spec, function, kwargs.pop("description", ""), frozenset(kwargs.pop("permissions", {"safe"})), kwargs.pop("risk", 0.0), kwargs.pop("cost", 1.0), kwargs.pop("confirmation", False), kwargs)
        if spec.name in self.tools: raise ValueError(f"tool already registered: {spec.name}")
        self.tools[spec.name]=spec
    def names(self): return sorted(self.tools)
    def describe(self, name=None):
        specs=self.tools.values() if name is None else [self.tools[name]]
        return [{"name":s.name,"description":s.description,"permissions":sorted(s.permissions),"risk":s.risk,"cost":s.cost,"confirmation":s.confirmation} for s in specs]
    def execute(self, name, arguments=None, *, approved=False):
        if name not in self.tools: return ToolResult(False,name,error="tool not registered")
        spec=self.tools[name]
        if spec.confirmation and not approved:
            self.audit.record(ToolAuditEvent(name,"execute",False,reason="confirmation required"))
            return ToolResult(False,name,error="confirmation required")
        if self.policy:
            decision=self.policy.check_tool(spec, arguments or {})
            if not decision.allowed:
                self.audit.record(ToolAuditEvent(name,"execute",False,reason=decision.reason))
                return ToolResult(False,name,error=decision.reason)
        started=monotonic()
        try:
            output=spec.function(**(arguments or {}))
            result=ToolResult(True,name,output=output,duration=monotonic()-started)
            self.audit.record(ToolAuditEvent(name,"execute",True,metadata={"duration":result.duration}))
            return result
        except Exception as exc:
            result=ToolResult(False,name,error=f"{type(exc).__name__}: {exc}",duration=monotonic()-started)
            self.audit.record(ToolAuditEvent(name,"execute",False,reason=result.error or "error"))
            return result
