import numpy as np
from models.transformer import Transformer
from training.lm_trainer import LMTrainer
from .contracts import TrainingJob
class TinySequenceDataset:
    def __init__(self,vocab_size=16,seq_len=4,seed=42):
        rng=np.random.default_rng(seed); self.x=rng.integers(0,vocab_size,size=(32,seq_len),dtype=np.int64); self.y=np.roll(self.x,-1,axis=1)
    def __len__(self): return len(self.x)
    def get_batch(self,batch_size=4,shuffle=True):
        idx=np.random.randint(0,len(self.x),size=batch_size) if shuffle else np.arange(batch_size)%len(self.x); return self.x[idx],self.y[idx]
class LocalTrainingRunner:
    def run_smoke(self,job:TrainingJob,steps=1):
        hp=job.hyperparameters; vocab=int(hp.get("vocab_size",16)); model=Transformer(vocab_size=vocab,d_model=int(hp.get("d_model",16)),num_heads=int(hp.get("num_heads",2)),d_ff=int(hp.get("d_ff",32)),num_layers=int(hp.get("num_layers",1)),max_len=int(hp.get("max_len",4)))
        ds=TinySequenceDataset(vocab,model.max_len,job.seed); trainer=LMTrainer(model,learning_rate=float(hp.get("lr",1e-3))); hist=trainer.train(ds,None,epochs=1,batch_size=min(2,len(ds)),batches_per_epoch=steps,verbose=False); job.status="completed"; job.metrics={"train_loss":hist["train_loss"][-1],"steps":steps}; return model,trainer,job
