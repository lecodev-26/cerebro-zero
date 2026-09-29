from dataclasses import asdict
from pathlib import Path
import hashlib,json,uuid
from .factory import DatasetValidator
from .contracts import TrainingJob
class TrainingPipeline:
    def __init__(self,artifact_root="~/.cerebro-zero/training"):
        self.root=Path(artifact_root).expanduser(); self.root.mkdir(parents=True,exist_ok=True); self.validator=DatasetValidator()
    def prepare(self,records):
        clean=self.validator.deduplicate(self.validator.validate(records)); split=int(len(clean)*0.8); return {"train":clean[:split],"validation":clean[split:],"count":len(clean)}
    def create_job(self,model_base,dataset_version,tokenizer,hyperparameters,seed=42,hardware="cpu"): return TrainingJob(uuid.uuid4().hex[:12],model_base,dataset_version,tokenizer,hyperparameters,seed,hardware)
    def record(self,job,metrics,checkpoint_path=None):
        job.status="completed"; job.metrics=dict(metrics)
        if checkpoint_path: job.checkpoint_hash=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
        path=self.root/f"{job.job_id}.json"; path.write_text(json.dumps(asdict(job),indent=2,default=str),encoding="utf-8"); return path
    def gate(self,metrics,previous=None,minimum_score=0.0):
        score=float(metrics.get("score",metrics.get("accuracy",0.0)))
        return score>=minimum_score and (previous is None or score>=float(previous.get("score",previous.get("accuracy",0.0))))
