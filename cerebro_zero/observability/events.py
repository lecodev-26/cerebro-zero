from dataclasses import dataclass, field
import time, uuid

@dataclass
class Event:
    name: str
    execution_id: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class EventTrace:
    def __init__(self, execution_id=None): self.execution_id=execution_id or uuid.uuid4().hex; self.events=[]
    def emit(self, name, **data):
        e=Event(name, self.execution_id, data); self.events.append(e); return e
