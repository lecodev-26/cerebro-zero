"""
Memoria episódica (Episodic Memory)
"""

import numpy as np
from collections import deque
from datetime import datetime

class EpisodicMemory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.episodes = deque(maxlen=max_size)
    
    def add(self, episode):
        """
        Añade un episodio a la memoria
        episode: {'input': ..., 'output': ..., 'context': ..., 'timestamp': ...}
        """
        if 'timestamp' not in episode:
            episode['timestamp'] = datetime.now().isoformat()
        self.episodes.append(episode)
    
    def get_recent(self, n=5):
        """
        Obtiene los últimos n episodios
        """
        return list(self.episodes)[-n:]
    
    def get_by_context(self, context, n=5):
        """
        Obtiene episodios por contexto
        """
        results = []
        for ep in self.episodes:
            if ep.get('context') == context:
                results.append(ep)
        return results[-n:]
    
    def get_all(self):
        return list(self.episodes)
    
    def clear(self):
        self.episodes.clear()
    
    def size(self):
        return len(self.episodes)
    
    def __repr__(self):
        return f"EpisodicMemory(size={len(self.episodes)}, max={self.max_size})"

def prueba_episodic():
    print("🧠 PROBANDO MEMORIA EPISÓDICA")
    print("="*30)
    
    memory = EpisodicMemory(max_size=5)
    
    # Añadir episodios
    memory.add({'input': '¿Cómo estás?', 'output': 'Bien, ¿y tú?', 'context': 'saludo'})
    memory.add({'input': '¿Qué hora es?', 'output': 'Son las 10:00', 'context': 'tiempo'})
    memory.add({'input': 'Hola', 'output': 'Hola', 'context': 'saludo'})
    
    print(f"📋 Últimos 2: {memory.get_recent(2)}")
    print(f"📋 Contexto 'saludo': {memory.get_by_context('saludo')}")
    
    memory.clear()
    print(f"🧹 Limpiado: {memory.size()}")
    
    print("\n✅ MEMORIA EPISÓDICA FUNCIONANDO!")

if __name__ == "__main__":
    prueba_episodic()
