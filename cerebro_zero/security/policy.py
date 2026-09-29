from dataclasses import dataclass
from v5.security import ActionGuard

@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    reason: str = ""

class SecurityPolicy:
    def __init__(self, max_actions=32, max_risk=1.0, permissions=None):
        self.guard=ActionGuard(max_actions,max_risk)
        self.permissions=set(permissions or {"safe"})
        self.actions=0
        self.max_actions=max_actions

    def validate(self, plan): return self.guard.validate(plan)

    def check_tool(self, spec, arguments=None):
        if self.actions >= self.max_actions:
            return ToolDecision(False,"action budget exceeded")
        if spec.risk > self.guard.max_risk:
            return ToolDecision(False,"tool risk exceeds policy")
        if spec.permissions and not set(spec.permissions).issubset(self.permissions):
            return ToolDecision(False,"required tool permission not granted")
        self.actions += 1
        return ToolDecision(True)

    def reset_budget(self): self.actions=0
