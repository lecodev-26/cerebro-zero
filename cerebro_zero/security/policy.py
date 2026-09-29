from v5.security import ActionGuard

class SecurityPolicy:
    def __init__(self, max_actions=32, max_risk=1.0):
        self.guard = ActionGuard(max_actions, max_risk)
    def validate(self, plan): return self.guard.validate(plan)
