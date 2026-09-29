import argparse,json,time
from v5 import CerebroZeroV5

def run_benchmark(cycles=2):
 a=CerebroZeroV5();t=time.perf_counter();goals=["solve a multi-step objective","fix and verify a failure"]
 cs=[a.run(goals[i%2]) for i in range(cycles)];dt=time.perf_counter()-t;s=sum(c.evaluation.success for c in cs)
 return {"version":a.version,"cycles":len(cs),"successful_cycles":s,"success_rate":s/len(cs) if cs else 1.0,"elapsed_seconds":dt,"cycles_per_second":len(cs)/dt if dt else 0.0,"stats":a.stats()}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--cycles",type=int,default=2);print(json.dumps(run_benchmark(max(0,p.parse_args().cycles)),indent=2))
