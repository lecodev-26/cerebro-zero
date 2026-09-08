"""
Memoria a corto plazo (Short-Term Memory)
"""

import numpy as np
from collections import deque

class ShortTermMemory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        self.embeddings = deque(maxlen=max_size)
    
    def add(self, item, embedding=None):
        """
        Añade un item a la memoria
        """
        self.memory.append(item)
        if embedding is not None:
            self.embeddings.append(embedding)
    
    def get_recent(self, n=5):
        """
        Obtiene los últimos n items
        """
        return list(self.memory)[-n:]
    
    def get_all(self):
        """
        Obtiene todos los items
        """
        return list(self.memory)
    
    def clear(self):
        """
        Limpia la memoria
        """
        self.memory.clear()
        self.embeddings.clear()
    
    def size(self):
        return len(self.memory)
    
    def is_full(self):
        return len(self.memory) >= self.max_size
    
    def __repr__(self):
        return f"ShortTermMemory(size={len(self.memory)}, max={self.max_size})"

def prueba_short_term():
    print("🧠 PROBANDO MEMORIA A CORTO PLAZO")
    print("="*30)
    
    memory = ShortTermMemory(max_size=5)
    
    for i in range(10):
        memory.add(f"item_{i}")
        print(f"   Añadido: item_{i} | Tamaño: {memory.size()}")
    
    print(f"\n📋 Últimos 3: {memory.get_recent(3)}")
    print(f"📋 Todos: {memory.get_all()}")
    
    memory.clear()
    print(f"🧹 Limpiado: {memory.size()}")
    
    print("\n✅ MEMORIA A CORTO PLAZO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_short_term()
