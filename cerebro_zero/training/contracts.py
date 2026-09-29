from dataclasses import dataclass,field
import hashlib,time
@dataclass(frozen=True)
class DatasetRecord:
    instruction:str; response:str; task:str="general"; source:str="unknown"; license:str="unknown"; provenance:str=""; metadata:dict=field(default_factory=dict)
@dataclass
class TrainingJob:
    job_id:str; model_base:str; dataset_version:str; tokenizer:str; hyperparameters:dict; seed:int=42; hardware:str="cpu"; status:str="pending"; metrics:dict=field(default_factory=dict); checkpoint_hash:str|None=None; created_at:float=field(default_factory=time.time)
    def fingerprint(self): return hashlib.sha256(repr((self.model_base,self.dataset_version,self.tokenizer,self.hyperparameters,self.seed,self.hardware)).encode()).hexdigest()
