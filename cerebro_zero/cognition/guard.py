from v5.security import ActionGuard

class Guard:
    def __init__(self, max_actions=32, max_risk=1.0): self._guard=ActionGuard(max_actions,max_risk)
    def validate(self, plan): return self._guard.validate(plan)
