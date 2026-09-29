from dataclasses import dataclass
import os, platform
@dataclass(frozen=True)
class ComputeInfo:
    backend:str; device:str; gpu:bool; workers:int
class ComputeBackend:
    def info(self): raise NotImplementedError
class LocalCPUBackend(ComputeBackend):
    def info(self): return ComputeInfo("cpu",platform.machine(),False,max(1,os.cpu_count() or 1))
class AutoBackend(ComputeBackend):
    def __init__(self): self.backend=LocalCPUBackend()
    def info(self): return self.backend.info()
