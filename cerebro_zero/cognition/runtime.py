from dataclasses import dataclass, field
import time
from v5.runtime import CerebroZeroV5
from ..config import Settings
from ..models import LocalEchoProvider
from ..observability import EventTrace
from ..core import PersistenceStore

@dataclass
class RunResult:
    text: str; execution_id: str; cycle_id: str; model: str; success: bool; score: float
    events: list = field(default_factory=list); cycle: object = None; metadata: dict = field(default_factory=dict)

class CognitiveRuntime:
    """Unified OBSERVE→UNDERSTAND→RETRIEVE→REASON→PLAN→GUARD→ACT→VERIFY→LEARN→CONSOLIDATE loop."""
    STAGES=("observe","understand","retrieve","reason","plan","guard","act","verify","learn","consolidate")
    def __init__(self, settings=None, provider=None):
        self.settings=settings or Settings.from_env(); self.provider=provider or LocalEchoProvider()
        self.engine=CerebroZeroV5(max_actions=self.settings.max_actions,max_risk=self.settings.max_risk)
        self.persistence=PersistenceStore(self.settings.data_dir)
    def run(self, goal, observation=None, constraints=None):
        trace=EventTrace(); started=time.time(); trace.emit("request.started",goal=goal)
        for stage in self.STAGES[:7]: trace.emit(stage)
        cycle=self.engine.run(goal,observation=observation,constraints=constraints)
        trace.emit("verify",success=bool(cycle.evaluation and cycle.evaluation.success))
        trace.emit("learn",lessons=list(cycle.lessons)); self.engine.consolidate(); trace.emit("consolidate")
        response=self.provider.generate([{"role":"user","content":goal}])
        score=float(cycle.evaluation.score if cycle.evaluation else 0.0)
        trace.emit("response.generated",model=response.model,score=score); trace.emit("request.completed",duration=time.time()-started)
        events=[e.__dict__ for e in trace.events]
        result=RunResult(response.text,cycle.id,cycle.id,response.model,bool(cycle.evaluation and cycle.evaluation.success),score,events,cycle,{"stage_count":len(self.STAGES),"profile":self.settings.profile})
        self.persistence.save_execution(result.execution_id,started,{"goal":goal,"score":score,"events":events})
        return result
    def close(self): return None
    def stats(self): return {**self.engine.stats(),"persisted_executions":self.persistence.count()}
