from dataclasses import dataclass, field
import time, uuid

@dataclass
class CycleBudget:
    token_budget: int=4096; action_budget: int=32; tool_budget: int=16; memory_budget: int=8; retry_budget: int=2

@dataclass
class CognitiveCycle:
    goal: str
    execution_id: str=field(default_factory=lambda:uuid.uuid4().hex)
    cycle_id: str=field(default_factory=lambda:uuid.uuid4().hex)
    observation: object=None; memories:list=field(default_factory=list); hypotheses:list=field(default_factory=list)
    plan:object=None; actions:list=field(default_factory=list); result:object=None; evaluation:object=None
    lessons:list=field(default_factory=list); started_at:float=field(default_factory=time.time); finished_at:float=None
