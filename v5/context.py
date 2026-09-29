from dataclasses import dataclass,field
import time
@dataclass
class Context:
 goal:str
 constraints:list=field(default_factory=list)
 observations:list=field(default_factory=list)
 memories:list=field(default_factory=list)
 metadata:dict=field(default_factory=dict)
 created:float=field(default_factory=time.time)
class ContextEngine:
 def build(self,goal,observation=None,memories=None,constraints=None):
  return Context(goal,constraints or [],[] if observation is None else [observation],memories or [],{"token_budget":4096})
 def compress(self,c,max_items=8):
  c.memories=c.memories[:max_items];c.observations=c.observations[:max_items];return c
