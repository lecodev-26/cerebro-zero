from dataclasses import dataclass, field

@dataclass
class CognitiveContext:
    goal: str
    observation: object = None
    memories: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

class ContextEngine:
    def build(self, goal, observation=None, memories=None, constraints=None):
        return CognitiveContext(goal, observation, memories or [], constraints or [], {"token_budget":4096})
    def compress(self, context, max_items=8):
        context.memories=context.memories[:max_items]
        return context
