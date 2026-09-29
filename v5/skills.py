from dataclasses import dataclass
@dataclass
class Skill:
 name:str
 steps:list
 uses:int=0
 success_rate:float=0.
class SkillRegistry:
 def __init__(self):self.skills={}
 def learn(self,name,steps,success):
  s=self.skills.get(name)
  if s is None:s=Skill(name,list(steps));self.skills[name]=s
  s.steps=list(steps);s.uses+=1;s.success_rate=((s.success_rate*(s.uses-1))+float(success))/s.uses;return s
 def select(self,names):
  return max((self.skills[n] for n in names if n in self.skills),key=lambda x:x.success_rate,default=None)
 def stats(self):return {"skills":len(self.skills),"uses":sum(s.uses for s in self.skills.values())}
