"""
Replay Buffer para aprendizaje continuo
Guarda experiencias antiguas para evitar olvido catastrófico.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional


class ReplayBuffer:
    """
    Buffer circular de experiencias.
    
    Uso:
        buffer = ReplayBuffer(max_size=1000)
        buffer.add(input_data, target_data, metadata={'task': 'math'})
        batch = buffer.sample(batch_size=32)
    """
    
    def __init__(self, max_size: int = 1000, seed: Optional[int] = None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(self, input_data: np.ndarray, target_data: np.ndarray, metadata: dict = None):
        """
        Añade una experiencia al buffer.
        
        input_data: (seq_len,) o (batch, seq_len)
        target_data: (seq_len,) o (batch, seq_len)
        metadata: información adicional (tarea, contexto, etc.)
        """
        # Normalizar a 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            target_data = target_data.reshape(1, -1)
        
        entry = {
            'input': input_data.copy(),
            'target': target_data.copy(),
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        }
        self.buffer.append(entry)
        self.total_added += 1
    
    def sample(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samplea un batch aleatorio del buffer.
        Combina las entradas en un solo batch.
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer vacío")
        
        # Samplear índices
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        
        inputs = []
        targets = []
        for i in indices:
            entry = self.buffer[i]
            for j in range(entry['input'].shape[0]):
                inputs.append(entry['input'][j])
                targets.append(entry['target'][j])
        
        # Limitar al batch_size
        inputs = inputs[:batch_size]
        targets = targets[:batch_size]
        
        return np.stack(inputs), np.stack(targets)
    
    def sample_by_task(self, task: str, batch_size: int = 32) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Samplea solo de una tarea específica"""
        matching = [e for e in self.buffer if e['metadata'].get('task') == task]
        if not matching:
            return None
        
        indices = random.sample(range(len(matching)), min(batch_size, len(matching)))
        inputs = []
        targets = []
        for i in indices:
            entry = matching[i]
            for j in range(entry['input'].shape[0]):
                inputs.append(entry['input'][j])
                targets.append(entry['target'][j])
        
        inputs = inputs[:batch_size]
        targets = targets[:batch_size]
        return np.stack(inputs), np.stack(targets)
    
    def size(self) -> int:
        return len(self.buffer)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def clear(self):
        self.buffer.clear()
        self.total_added = 0
    
    def tasks(self) -> List[str]:
        """Lista las tareas presentes en el buffer"""
        tasks = set()
        for entry in self.buffer:
            task = entry['metadata'].get('task')
            if task:
                tasks.add(task)
        return sorted(tasks)
    
    def stats(self) -> dict:
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'tasks': self.tasks(),
        }
    
    def __repr__(self):
        return f"ReplayBuffer({len(self.buffer)}/{self.max_size})"


if __name__ == "__main__":
    print("🧪 PROBANDO REPLAY BUFFER")
    print("="*50)
    
    buffer = ReplayBuffer(max_size=100, seed=42)
    
    # Añadir experiencias
    print("\n📝 Añadiendo experiencias:")
    for i in range(10):
        X = np.random.randint(0, 100, (4, 8))
        y = np.random.randint(0, 100, (4, 8))
        buffer.add(X, y, {'task': 'math' if i % 2 == 0 else 'lang'})
    
    print(f"   {buffer}")
    print(f"   Stats: {buffer.stats()}")
    
    # Samplear
    print("\n🔍 Sampleando batch:")
    X, y = buffer.sample(batch_size=8)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Samplear por tarea
    print("\n🔍 Sampleando por tarea 'math':")
    result = buffer.sample_by_task('math', batch_size=4)
    if result:
        X, y = result
        print(f"   X shape: {X.shape}")
    
    print("\n✅ REPLAY BUFFER FUNCIONANDO")
