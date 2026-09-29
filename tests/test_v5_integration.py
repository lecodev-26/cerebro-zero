from v5 import CerebroZeroV5
from v5.security import SecurityError

def test_v5_integrates_context_guard_skills_evaluator():
    c=CerebroZeroV5();x=c.run("solve objective")
    assert x.evaluation.success
    assert c.skills.stats()["skills"] == 1
    assert c.memory.procedural
    assert c.evaluator.score(x) == x.evaluation.score

def test_v5_executes_registered_tool():
    c=CerebroZeroV5();c.planner.make_plan=lambda g,h: __import__('v5.runtime',fromlist=['Plan','Action']).Plan(g,[__import__('v5.runtime',fromlist=['Action']).Action('add',{'a':2,'b':3})],.9,1,.0)
    assert c.run("add") .actions[0]["result"] == 5

def test_v5_guard_rejects_excessive_plan():
    c=CerebroZeroV5(max_actions=1)
    from v5.runtime import Plan, Action
    c.planner.make_plan=lambda g,h: Plan(g,[Action('a'),Action('b')],.5,2,.1)
    try: c.run("too many")
    except SecurityError: return
    assert False
