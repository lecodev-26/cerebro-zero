from dataclasses import dataclass,field
@dataclass(frozen=True)
class TeacherSpec:
    name:str; provider:str; capabilities:tuple[str,...]=(); license:str="unknown"; terms:str=""; metadata:dict=field(default_factory=dict)
class TeacherRegistry:
    def __init__(self): self._teachers={}
    def register(self,spec): self._teachers[spec.name]=spec
    def get(self,name): return self._teachers[name]
    def list(self): return list(self._teachers.values())
