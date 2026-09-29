from dataclasses import dataclass,field
@dataclass
class Metrics:
    counters:dict=field(default_factory=dict); timings:dict=field(default_factory=dict); gauges:dict=field(default_factory=dict)
    def inc(self,name,value=1): self.counters[name]=self.counters.get(name,0)+value
    def observe(self,name,value): self.timings.setdefault(name,[]).append(float(value))
    def set(self,name,value): self.gauges[name]=value
    def snapshot(self): return {"counters":dict(self.counters),"timings":{k:list(v) for k,v in self.timings.items()},"gauges":dict(self.gauges)}
