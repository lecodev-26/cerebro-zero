"""
Entrenador de Language Model con BACKPROP REAL
Usa loss.backward() y optimizer.step() — sin aproximaciones.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import time
from core.tensor import Tensor
from core.optimizers import Adam
from models.transformer import Transformer


class LMTrainer:
    """
    Entrenador de Language Model con backprop real.
    
    Uso:
        trainer = LMTrainer(model, learning_rate=0.01)
        history = trainer.train(train_ds, val_ds, epochs=10, batch_size=8)
    """
    
    def __init__(self, model: Transformer, learning_rate: float = 0.01):
        self.model = model
        self.lr = learning_rate
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
        }
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.best_weights = None
    
    # ============================================
    # LOSS Y GRADIENTES
    # ============================================
    
    def cross_entropy_loss(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Cross-entropy loss dentro del grafo.
        
        logits: Tensor (batch, seq_len, vocab_size)
        targets: np.ndarray (batch, seq_len)
        
        Loss = -mean(log(softmax(logits)[target]))
        
        Implementación:
        - Aplicar softmax sobre logits
        - Codificar targets como one-hot
        - Loss = -sum(target_onehot * log(probs))
        """
        batch, seq_len, vocab_size = logits.data.shape
        targets_flat = targets.reshape(-1).astype(int)
        batch_seq = batch * seq_len
        
        # Softmax con estabilidad numérica
        probs = logits.softmax(axis=-1)
        
        # Clip para evitar log(0)
        eps = 1e-10
        probs_clipped = probs + eps
        
        # Log de las probabilidades
        log_probs = probs_clipped.log()  # (batch, seq, vocab)
        
        # Crear one-hot de los targets
        targets_onehot = np.zeros((batch, seq_len, vocab_size), dtype=log_probs.data.dtype)
        for b in range(batch):
            for s in range(seq_len):
                targets_onehot[b, s, targets[b, s]] = 1.0
        
        # Loss = -mean(sum(target_onehot * log_probs, axis=-1))
        # Equivalente a: -sum(target_onehot * log_probs) / batch_seq
        # Usamos multiplicación elemento a elemento + sum
        
        # Multiplicar elemento a elemento
        product = log_probs * Tensor(targets_onehot, requires_grad=False)  # (batch, seq, vocab)
        
        # Sumar sobre vocab
        summed = product.sum(axis=-1)  # (batch, seq)
        
        # Negativo y media
        loss = -summed.mean()
        
        return loss
    
    # ============================================
    # TRAIN STEP
    # ============================================
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Un paso de entrenamiento con backprop REAL.
        
        X: (batch, seq_len) — tokens de entrada
        y: (batch, seq_len) — tokens objetivo
        """
        # Forward
        x_t = Tensor(X.astype(int))
        logits = self.model.forward(x_t)
        
        # Loss
        loss = self.cross_entropy_loss(logits, y)
        
        # Backward REAL (sin aproximaciones)
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return float(loss.data)
    
    # ============================================
    # EVALUACIÓN
    # ============================================
    
    def evaluate(self, dataset, batch_size: int = 8, num_batches: int = 10) -> dict:
        """Evalúa el modelo sobre un dataset"""
        losses = []
        for _ in range(num_batches):
            X, y = dataset.get_batch(batch_size=batch_size, shuffle=True)
            x_t = Tensor(X.astype(int))
            logits = self.model.forward(x_t)
            loss = self.cross_entropy_loss(logits, y)
            losses.append(float(loss.data))
        
        avg_loss = float(np.mean(losses))
        ppl = float(np.exp(min(avg_loss, 20)))  # Cap para evitar overflow
        
        return {'loss': avg_loss, 'ppl': ppl}
    
    # ============================================
    # ENTRENAMIENTO
    # ============================================
    
    def train(self, train_ds, val_ds=None, epochs: int = 10,
              batch_size: int = 8, batches_per_epoch: int = 20,
              verbose: bool = True, eval_every: int = 1,
              early_stopping_patience: int = None):
        """
        Bucle de entrenamiento completo con backprop REAL.
        """
        if verbose:
            print(f"🏋️ Entrenando {self.model.name}")
            print(f"   LR: {self.lr}")
            print(f"   Épocas: {epochs}")
            print(f"   Batches/época: {batches_per_epoch}")
            print(f"   Batch size: {batch_size}")
            print(f"   Train samples: {len(train_ds)}")
            if val_ds:
                print(f"   Val samples: {len(val_ds)}")
            print("="*60)
        
        start_time = time.time()
        epochs_no_improvement = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_losses = []
            
            for _ in range(batches_per_epoch):
                X, y = train_ds.get_batch(batch_size=batch_size, shuffle=True)
                loss = self.train_step(X, y)
                train_losses.append(loss)
            
            avg_train_loss = float(np.mean(train_losses))
            train_ppl = float(np.exp(min(avg_train_loss, 20)))
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_ppl'].append(train_ppl)
            
            val_info = ""
            improved = False
            if val_ds is not None and (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(val_ds, batch_size=batch_size, num_batches=5)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_ppl'].append(val_metrics['ppl'])
                val_info = f" | val_loss={val_metrics['loss']:.4f}, val_ppl={val_metrics['ppl']:.2f}"
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_epoch = epoch
                    # Guardar mejor peso
                    self.best_weights = [p.data.copy() for p in self.model.parameters()]
                    improved = True
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
            
            epoch_time = time.time() - epoch_start
            if verbose:
                marker = " ⭐" if improved else ""
                print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f}, "
                      f"train_ppl={train_ppl:.2f}{val_info} | {epoch_time:.2f}s{marker}")
            
            # Early stopping
            if early_stopping_patience is not None and epochs_no_improvement >= early_stopping_patience:
                if verbose:
                    print(f"⏹️ Early stopping: {early_stopping_patience} épocas sin mejora")
                break
        
        total_time = time.time() - start_time
        if verbose:
            print("="*60)
            print(f"✅ Entrenamiento completado en {total_time:.2f}s")
            if self.best_epoch >= 0:
                print(f"   Mejor época: {self.best_epoch+1}")
                print(f"   Mejor val_loss: {self.best_val_loss:.4f}")
                # Restaurar mejores pesos
                if self.best_weights is not None:
                    for p, w in zip(self.model.parameters(), self.best_weights):
                        p.data = w.copy()
                    if verbose:
                        print(f"   ✅ Mejores pesos restaurados")
        
        self.model._trained = True
        return self.history
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def save(self, path: str):
        """Guarda el modelo y el histórico"""
        state = {
            'model_weights': [p.data.copy() for p in self.model.parameters()],
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'num_heads': self.model.num_heads,
            'num_layers': self.model.num_layers,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Modelo guardado en {path}")
    
    def load(self, path: str):
        """Carga el modelo y el histórico"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        params = self.model.parameters()
        for p, w in zip(params, state['model_weights']):
            p.data = w.copy()
        
        self.history = state.get('history', self.history)
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.best_epoch = state.get('best_epoch', -1)
        print(f"📂 Modelo cargado desde {path}")


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO LM TRAINER V3.0 (backprop REAL)")
    print("="*60)
    
    from language.tokenizer import Tokenizer
    from language.dataset import TextDataset
    
    # Cargar datos
    all_texts = []
    for split in ['train', 'validation', 'test']:
        for prefix in ['', '../']:
            path = f"{prefix}datasets/{split}.txt"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
                break
    
    tok = Tokenizer()
    tok.build_vocab(all_texts, vocab_size=200)
    
    def find_path(name):
        for prefix in ['', '../']:
            p = f"{prefix}datasets/{name}.txt"
            if os.path.exists(p):
                return p
        return None
    
    train_ds = TextDataset(find_path('train'), tok, context_len=8)
    val_ds = TextDataset(find_path('validation'), tok, context_len=8)
    
    # Modelo más pequeño para móvil
    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=1,
        max_len=8,
    )
    model.summary()
    
    trainer = LMTrainer(model, learning_rate=0.01)
    
    print("\n🏋️ ENTRENANDO CON BACKPROP REAL...\n")
    history = trainer.train(
        train_ds, val_ds,
        epochs=20,
        batch_size=8,
        batches_per_epoch=10,
        verbose=True,
        early_stopping_patience=5,
    )
    
    print("\n📊 Estadísticas:")
    if history['train_loss']:
        print(f"   Loss inicial: {history['train_loss'][0]:.4f}")
        print(f"   Loss final:   {history['train_loss'][-1]:.4f}")
        mejora = (history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0]
        print(f"   Mejora:       {mejora*100:.1f}%")
    
    print("\n🔮 Generación de ejemplo:")
    prompt = tok.encode("la inteligencia")
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8)
    texto = tok.decode(generated)
    print(f"   Prompt: 'la inteligencia'")
    print(f"   Generado: {texto}")
    
    print("\n✅ LM TRAINER V3.0 FUNCIONANDO CON BACKPROP REAL")
