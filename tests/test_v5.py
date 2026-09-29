from v5 import CerebroZeroV5
def test_cycle():
 a=CerebroZeroV5();c=a.run("learn and verify");assert c.evaluation.success and a.memory.stats()["episodic"]==1
def test_tool():
 a=CerebroZeroV5();assert a.tools.execute("add",{"a":2,"b":3})==5
def test_repair():
 a=CerebroZeroV5()
 def bad(x):
  if x.name=="execute_goal": raise RuntimeError("boom")
  return "ok"
 c=a.run("repair",executor=bad);assert c.plan.repaired and not c.evaluation.success
def test_retrieval():
 a=CerebroZeroV5();a.memory.add_semantic("NumPy transformer memory");assert a.memory.retrieve("transformer memory")
def test_learning():
 a=CerebroZeroV5();a.run("test");assert a.learning.strategies["default"]["uses"]==1
