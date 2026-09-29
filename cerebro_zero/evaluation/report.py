from dataclasses import dataclass,asdict,field
import json,time
@dataclass
class EvaluationReport:
    model:str; version:str; metrics:dict; benchmark:str; passed:bool; timestamp:float=field(default_factory=time.time)
    def to_dict(self): return asdict(self)
    def save(self,path):
        with open(path,"w",encoding="utf-8") as f: json.dump(self.to_dict(),f,indent=2)
        return path
