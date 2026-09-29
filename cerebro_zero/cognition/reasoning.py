from ..compat import ReasoningEngine

class Reasoner(ReasoningEngine):
    """1.0 boundary around the proven reasoning primitive."""
    def reason(self, goal, memories): return self.infer(goal, memories)
