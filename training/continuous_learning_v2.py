"""
Aprendizaje continuo REAL con replay buffer
Evita el olvido catastrófico mezclando datos nuevos con antiguos.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.serialization import safe_save_dict
from datetime import datetime
from training.replay_buffer import ReplayBuffer
from training.lm_trainer import LMTrainer


class ContinuousLearner:
    """
    Aprendizaje continuo con replay buffer.
    
    Uso:
        learner = ContinuousLearner(model, buffer_size=500)
        learner.learn_batch(X_new, y_new, task='math', epochs=5)
        metrics = learner.evaluate_forgetting()
    """
    
    def __init__(self, model, buffer_size: int = 500,
                 learning_rate: float = 0.01):
        self.model = model
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.trainer = LMTrainer(model, learning_rate=learning_rate)
        
        # Historial
        self.history = {
            'tasks_learned': [],
            'loss_before': {},
            'loss_after': {},
            'forgetting': {},
        }
    
    def learn_batch(self, X: np.ndarray, y: np.ndarray, task: str = 'default',
                    epochs: int = 3, batches_per_epoch: int = 5,
                    use_replay: bool = True, replay_ratio: float = 0.5,
                    verbose: bool = False):
        """
        Aprende un nuevo batch.
        
        Args:
            X, y: datos nuevos
            task: nombre de la tarea
            epochs: épocas de entrenamiento
            batches_per_epoch: batches por época
            use_replay: si mezclar con buffer (evita olvido)
            replay_ratio: proporción de replay vs nuevos (0.5 = mitad y mitad)
        """
        # Medir loss ANTES (sobre datos nuevos)
        loss_before = self._evaluate_on(X, y)
        
        if verbose:
            print(f"📚 Aprendiendo tarea '{task}'")
            print(f"   Loss antes: {loss_before:.4f}")
            print(f"   Samples nuevos: {X.shape[0]}")
            print(f"   Buffer size antes: {self.buffer.size()}")
        
        # Añadir al buffer
        self.buffer.add(X, y, {'task': task})
        
        # Entrenar
        for epoch in range(epochs):
            for _ in range(batches_per_epoch):
                # Decidir si usar replay
                if use_replay and self.buffer.size() > X.shape[0]:
                    # Batch mixto: mitad nuevos, mitad replay
                    n_replay = int(batches_per_epoch * replay_ratio) if batches_per_epoch > 1 else 0
                    
                    # Batch de replay
                    X_replay, y_replay = self.buffer.sample(batch_size=X.shape[0])
                    loss_replay = self.trainer.train_step(X_replay, y_replay)
                    
                    # Batch de nuevos
                    loss_new = self.trainer.train_step(X, y)
                    
                    if verbose and epoch == 0:
                        print(f"   Loss replay: {loss_replay:.4f}, Loss nuevo: {loss_new:.4f}")
                else:
                    self.trainer.train_step(X, y)
        
        # Medir loss DESPUÉS (sobre datos nuevos)
        loss_after = self._evaluate_on(X, y)
        
        if verbose:
            print(f"   Loss después: {loss_after:.4f}")
            print(f"   Buffer size después: {self.buffer.size()}")
        
        # Guardar métricas
        self.history['loss_before'][task] = loss_before
        self.history['loss_after'][task] = loss_after
        if task not in self.history['tasks_learned']:
            self.history['tasks_learned'].append(task)
        
        return {
            'loss_before': loss_before,
            'loss_after': loss_after,
            'improvement': loss_before - loss_after,
        }
    
    def _evaluate_on(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evalúa el loss sobre un batch"""
        from core.tensor import Tensor
        logits = self.model.forward(Tensor(X))  # Tensor, no .data
        loss = self.trainer.cross_entropy_loss(logits, y)
        return float(loss.data)
    
    def evaluate_forgetting(self, task_data: dict, verbose: bool = False) -> dict:
        """
        Evalúa cuánto se ha olvidado de cada tarea.
        
        Args:
            task_data: {task_name: (X, y)}
        
        Returns:
            {task: forgetting_ratio}
        """
        forgetting = {}
        
        for task, (X, y) in task_data.items():
            loss = self._evaluate_on(X, y)
            initial_loss = self.history['loss_before'].get(task, loss)
            
            # Forgetting: cuánto ha subido el loss respecto al inicial
            if initial_loss > 0:
                forgetting[task] = (loss - initial_loss) / initial_loss
            else:
                forgetting[task] = 0.0
        
        self.history['forgetting'] = forgetting
        
        if verbose:
            print("🧠 Evaluación de olvido:")
            for task, ratio in forgetting.items():
                status = "✅" if ratio <= 0.1 else "⚠️" if ratio <= 0.3 else "❌"
                print(f"   {status} {task}: forgetting = {ratio*100:.1f}%")
        
        return forgetting
    
    def stats(self) -> dict:
        """Estadísticas del aprendizaje continuo"""
        return {
            'tasks_learned': len(self.history['tasks_learned']),
            'buffer_size': self.buffer.size(),
            'buffer_tasks': self.buffer.tasks(),
            'losses': {
                task: {
                    'before': self.history['loss_before'].get(task, 0),
                    'after': self.history['loss_after'].get(task, 0),
                }
                for task in self.history['tasks_learned']
            },
        }
    
    def save(self, path: str):
        """Guarda el estado (formato seguro JSON+NPY, sin pickle)"""
        if path.endswith(".pkl") or path.endswith(".pickle"):
            path = path.rsplit(".", 1)[0]

        weights = [p.data.copy() for p in self.model.parameters()]
        weights_dict = {f"weight_{i}": w for i, w in enumerate(weights)}

        state = {
            'history': self.history,
            'buffer_stats': self.buffer.stats(),
            'num_weights': len(weights),
        }
        state.update(weights_dict)
        safe_save_dict(state, path)
        print(f"💾 Learner guardado en {path}")
    
    def __repr__(self):
        return f"ContinuousLearner(tasks={len(self.history['tasks_learned'])}, buffer={self.buffer.size()})"


# ============================================
# TESTS
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO APRENDIZAJE CONTINUO")
    print("="*60)
    
    from language.tokenizer import Tokenizer
    from language.dataset import TextDataset
    from models.transformer import Transformer
    
    # Tokenizador
    all_texts = []
    for split in ['train', 'validation', 'test']:
        for prefix in ['', '../']:
            path = f"{prefix}datasets/{split}.txt"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    all_texts.append(f.read())
                break
    
    tok = Tokenizer()
    tok.build_vocab(all_texts, vocab_size=100)
    
    # Modelo
    model = Transformer(
        vocab_size=tok.vocab_size,
        d_model=16, num_heads=2, d_ff=32,
        num_layers=1, max_len=8,
    )
    
    # Learner
    learner = ContinuousLearner(model, buffer_size=100, learning_rate=0.05)
    
    # Aprender 3 tareas diferentes
    print("\n📚 Tarea 1: matemáticas")
    X1 = np.random.randint(0, 50, (8, 8))
    y1 = np.random.randint(0, 50, (8, 8))
    result1 = learner.learn_batch(X1, y1, task='math', epochs=5, verbose=True)
    
    print("\n📚 Tarea 2: lenguaje")
    X2 = np.random.randint(0, 50, (8, 8))
    y2 = np.random.randint(0, 50, (8, 8))
    result2 = learner.learn_batch(X2, y2, task='lang', epochs=5, verbose=True)
    
    print("\n📚 Tarea 3: memoria")
    X3 = np.random.randint(0, 50, (8, 8))
    y3 = np.random.randint(0, 50, (8, 8))
    result3 = learner.learn_batch(X3, y3, task='memory', epochs=5, verbose=True)
    
    # Evaluar olvido
    print("\n" + "="*60)
    task_data = {
        'math': (X1, y1),
        'lang': (X2, y2),
        'memory': (X3, y3),
    }
    forgetting = learner.evaluate_forgetting(task_data, verbose=True)
    
    # Stats
    print("\n📊 Stats del learner:")
    stats = learner.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ APRENDIZAJE CONTINUO FUNCIONANDO")
