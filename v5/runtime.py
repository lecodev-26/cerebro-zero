from dataclasses import dataclass, field
from collections import deque
import re, time, uuid, random
from .context import ContextEngine
from .security import ActionGuard, SecurityError
from .skills import SkillRegistry
from .evaluation import Evaluator

@dataclass
class Evidence: content:str; source:str="unknown"; strength:float=1.0
@dataclass
class Hypothesis: statement:str; confidence:float; evidence:list=field(default_factory=list)
@dataclass
class ToolSpec: name:str; function:object; description:str=""; permissions:set=field(default_factory=lambda:{"safe"}); risk:float=0.; cost:float=1.
@dataclass
class Action: name:str; arguments:dict=field(default_factory=dict); expected:object=None; risk:float=0.
@dataclass
class Plan: goal:str; steps:list; confidence:float=.5; cost:float=0.; risk:float=0.; repaired:bool=False
@dataclass
class Evaluation: success:bool; score:float; prediction_error:float; confidence:float; lessons:list=field(default_factory=list)
@dataclass
class CognitiveCycle:
 goal:str; id:str=field(default_factory=lambda:uuid.uuid4().hex); observation:object=None; memories:list=field(default_factory=list); hypotheses:list=field(default_factory=list); plan:object=None; actions:list=field(default_factory=list); evaluation:object=None; lessons:list=field(default_factory=list); started_at:float=field(default_factory=time.time); finished_at:float=None
 def to_dict(self): return {"id":self.id,"goal":self.goal,"observation":self.observation,"memories":self.memories,"hypotheses":[h.__dict__ for h in self.hypotheses],"plan":self.plan.__dict__ if self.plan else None,"actions":self.actions,"evaluation":self.evaluation.__dict__ if self.evaluation else None,"lessons":self.lessons}

class MemorySystem:
 def __init__(self,working_capacity=16,episodic_capacity=1000): self.working={};self.capacity=working_capacity;self.episodic=deque(maxlen=episodic_capacity);self.semantic=[];self.procedural=[]
 def _tokens(self,s): return set(re.findall(r"[\w]+",str(s).lower()))
 def add_working(self,k,v,importance=.5):
  if k not in self.working and len(self.working)>=self.capacity: del self.working[min(self.working,key=lambda x:self.working[x]["importance"])]
  self.working[k]={"value":v,"importance":importance,"accesses":0,"time":time.time()}
 def remember_episode(self,c,outcome): self.episodic.append({"id":c.id,"goal":c.goal,"actions":c.actions,"lessons":c.lessons,"outcome":outcome,"time":time.time()})
 def add_semantic(self,text,metadata=None,importance=.5): self.semantic.append({"text":str(text),"metadata":metadata or {},"importance":importance,"time":time.time()})
 def add_procedure(self,name,steps,success):
  x=next((x for x in self.procedural if x["name"]==name),None)
  if x is None:self.procedural.append({"name":name,"steps":list(steps),"uses":1,"success_rate":float(success)})
  else:x["steps"]=list(steps);x["uses"]+=1;x["success_rate"]=(x["success_rate"]*(x["uses"]-1)+float(success))/x["uses"]
 def retrieve(self,q,k=5):
  qt=self._tokens(q);hits=[]
  for x in self.semantic:
   t=self._tokens(x["text"]);over=len(qt&t)/max(1,len(qt|t))
   if over:hits.append((.7*over+.3*x["importance"],{"kind":"semantic","text":x["text"],"metadata":x["metadata"],"score":over}))
  for x in self.episodic:
   t=self._tokens(x["goal"]+" "+" ".join(x["lessons"]));over=len(qt&t)/max(1,len(qt|t))
   if over:hits.append((over,{"kind":"episodic","text":x["goal"],"lessons":x["lessons"],"score":over}))
  hits.sort(reverse=True,key=lambda x:x[0]);return [x[1] for x in hits[:k]]
 def consolidate(self):
  added=0
  for e in self.episodic:
   for l in e["lessons"]: self.add_semantic(l,{"source":"consolidation","episode":e["id"]},.7);added+=1
  return added
 def stats(self): return {"working":len(self.working),"episodic":len(self.episodic),"semantic":len(self.semantic),"procedural":len(self.procedural)}

class ReasoningEngine:
 def __init__(self): self.history=[]
 def infer(self,goal,memories):
  hs=[Hypothesis("Decompose and verify the objective",.6,[Evidence(goal,"goal",.7)])]
  if memories: hs.insert(0,Hypothesis("Reuse relevant prior experience",.75,[Evidence(str(memories),"memory",.8)]))
  if any(w in goal.lower() for w in ("fix","debug","repair","error","failure")): hs.insert(0,Hypothesis("Diagnose before mutating state",.85,[Evidence("failure-oriented task","heuristic",.9)]))
  hs.sort(key=lambda x:x.confidence,reverse=True);self.history.append(hs);return hs

class Planner:
 def make_plan(self,goal,hypotheses): return Plan(goal,[Action("understand",{"goal":goal}),Action("reason",{"hypotheses":[h.statement for h in hypotheses[:3]]}),Action("execute_goal",{"goal":goal}),Action("verify",{"goal":goal})],.7,4,.1)
 def repair(self,plan,failed,reason):
  steps=[x for x in plan.steps if x.name!=failed];steps.insert(-1,Action("recover",{"failed_action":failed,"reason":reason}));return Plan(plan.goal,steps,max(.1,plan.confidence-.1),plan.cost+1,min(1,plan.risk+.1),True)

class PermissionPolicy:
 def __init__(self,allowed=None): self.allowed=allowed or {"safe"}
 def permits(self,s): return s.permissions.issubset(self.allowed)
class ToolRegistry:
 def __init__(self,policy=None): self.tools={};self.policy=policy or PermissionPolicy()
 def register(self,s): self.tools[s.name]=s
 def execute(self,name,args=None):
  s=self.tools[name]
  if not self.policy.permits(s): raise PermissionError(name)
  return s.function(**(args or {}))

class LearningEngine:
 def __init__(self,capacity=2000): self.experiences=[];self.capacity=capacity;self.strategies={};self.reward=0.
 def record(self,state,action,outcome,reward,verified):
  self.experiences.append({"state":state,"action":action,"outcome":outcome,"reward":reward,"verified":verified});self.experiences=self.experiences[-self.capacity:];self.reward+=reward
 def learn_strategy(self,name,steps,success):
  x=self.strategies.setdefault(name,{"steps":list(steps),"uses":0,"successes":0});x["steps"]=list(steps);x["uses"]+=1;x["successes"]+=int(success);x["success_rate"]=x["successes"]/x["uses"]
 def replay(self,k=32): return random.sample(self.experiences,min(k,len(self.experiences)))
 def stats(self): return {"experiences":len(self.experiences),"strategies":len(self.strategies),"reward":self.reward}

class CerebroZeroV5:
 version="5.0.0"
 def __init__(self,name="cerebro_v5",working_capacity=16,max_actions=32,max_risk=1.0):
  self.name=name;self.memory=MemorySystem(working_capacity);self.reasoning=ReasoningEngine();self.planner=Planner();self.tools=ToolRegistry();self.learning=LearningEngine();self.context=ContextEngine();self.guard=ActionGuard(max_actions,max_risk);self.skills=SkillRegistry();self.evaluator=Evaluator();self.cycles=[]
  self.tools.register(ToolSpec("echo",lambda text:str(text),"Safe text operation"));self.tools.register(ToolSpec("add",lambda a,b:a+b,"Safe arithmetic"))
 def run(self,goal,observation=None,executor=None,constraints=None):
  c=CognitiveCycle(goal);c.observation=observation if observation is not None else goal
  self.memory.add_working("goal",goal,1.);c.memories=self.memory.retrieve(goal)
  ctx=self.context.build(goal,c.observation,c.memories,constraints);self.context.compress(ctx)
  c.hypotheses=self.reasoning.infer(goal,c.memories);c.plan=self.planner.make_plan(goal,c.hypotheses);self.guard.validate(c.plan)
  skill=self.skills.select([goal,"default"]);ex=executor or self._default_executor
  for a in c.plan.steps:
   try:
    result=self.tools.execute(a.name,a.arguments) if a.name in self.tools.tools else ex(a)
    c.actions.append({"name":a.name,"arguments":a.arguments,"result":result,"success":True})
   except Exception as e:
    c.actions.append({"name":a.name,"arguments":a.arguments,"error":str(e),"success":False})
    c.plan=self.planner.repair(c.plan,a.name,str(e));self.guard.validate(c.plan);break
  score=sum(x["success"] for x in c.actions)/max(1,len(c.actions));success=all(x["success"] for x in c.actions) and bool(c.actions)
  c.lessons.append("verified success" if success else "failure requires recovery learning");c.evaluation=Evaluation(success,score,abs(c.plan.confidence-score),c.plan.confidence,c.lessons.copy())
  self.evaluator.score(c);self.learning.record({"goal":goal},c.plan.__dict__,{"score":score},score,success);self.learning.learn_strategy("default",[x.name for x in c.plan.steps],success);self.skills.learn("default",[x.name for x in c.plan.steps],success);self.memory.add_procedure("default",[x.name for x in c.plan.steps],success);self.memory.remember_episode(c,c.evaluation.__dict__);c.finished_at=time.time();self.cycles.append(c);return c
 def _default_executor(self,a): return {"status":"simulated","action":a.name}
 def consolidate(self): return self.memory.consolidate()
 def stats(self): return {"version":self.version,"cycles":len(self.cycles),"memory":self.memory.stats(),"learning":self.learning.stats(),"skills":self.skills.stats(),"tools":len(self.tools.tools)}
