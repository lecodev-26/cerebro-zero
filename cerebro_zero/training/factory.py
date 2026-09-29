from .contracts import DatasetRecord
class SyntheticDataFactory:
    def __init__(self,teacher_registry): self.teachers=teacher_registry
    def from_examples(self,examples,teacher=None):
        source=teacher or "curated"; names={t.name:t for t in self.teachers.list()}; spec=names.get(source); lic=spec.license if spec else "curated"
        return [DatasetRecord(e["instruction"],e["response"],e.get("task","general"),source,lic,e.get("provenance",source),e.get("metadata",{})) for e in examples]
    def generate(self,prompts,teacher_name,generate_fn):
        teacher=self.teachers.get(teacher_name)
        return [DatasetRecord(p,generate_fn(teacher,p),source=teacher.name,license=teacher.license,provenance=teacher.provider) for p in prompts]
    def from_teacher_results(self, results):
        return [DatasetRecord(r.prompt, r.response, source=r.teacher, license=r.metadata.get("license", "unknown"), provenance=r.metadata.get("provider", r.teacher), metadata={"model": r.model, **r.metadata}) for r in results]
class DatasetValidator:
    def validate(self,records,allowed_licenses=None):
        allowed=set(allowed_licenses or {"MIT","Apache-2.0","CC-BY","curated","provider-output"}); return [r for r in records if r.instruction.strip() and r.response.strip() and r.license in allowed]
    def deduplicate(self,records):
        seen=set(); out=[]
        for r in records:
            k=(r.instruction.strip().lower(),r.response.strip().lower())
            if k not in seen: seen.add(k); out.append(r)
        return out
