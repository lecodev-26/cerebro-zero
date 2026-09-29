class Evaluator:
 def score(self,cycle): return cycle.evaluation.score if cycle.evaluation else 0.
 def compare(self,before,after): return {"improvement":float(after-before),"before":float(before),"after":float(after)}
 def calibration_error(self,records):
  if not records:return 0.
  return sum(abs(r["confidence"]-r["actual"]) for r in records)/len(records)
