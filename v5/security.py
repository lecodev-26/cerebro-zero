class SecurityError(Exception): pass
class ActionGuard:
 def __init__(self,max_actions=32,max_risk=1.0): self.max_actions=max_actions;self.max_risk=max_risk
 def validate(self,plan):
  if len(plan.steps)>self.max_actions: raise SecurityError("action budget exceeded")
  if plan.risk>self.max_risk: raise SecurityError("risk budget exceeded")
  return True
