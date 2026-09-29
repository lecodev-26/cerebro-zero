import time,json
from v5 import CerebroZeroV5
def run_benchmark():
 a=CerebroZeroV5();t=time.perf_counter();cs=[a.run("solve a multi-step objective"),a.run("fix and verify a failure")];dt=time.perf_counter()-t;s=sum(c.evaluation.success for c in cs)
 return {"version":a.version,"cycles":len(cs),"successful_cycles":s,"success_rate":s/len(cs),"elapsed_seconds":dt,"stats":a.stats()}
if __name__=="__main__": print(json.dumps(run_benchmark(),indent=2))
