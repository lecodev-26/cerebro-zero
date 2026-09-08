"""
Sistema de recuperación de memoria
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory

class MemoryRetrieval:
    def __init__(self):
        self.short_term = ShortTermMemory(max_size=100)
        self.long_term = LongTermMemory(filename="memoria_largo_plazo.json")
        self.episodic = EpisodicMemory(max_size=100)
    
    def remember(self, query, limit=5):
        """
        Recupera información relevante de todas las memorias
        """
        results = {
            'short_term': [],
            'long_term': [],
            'episodic': []
        }
        
        # 1. Buscar en memoria a corto plazo
        recent = self.short_term.get_recent(limit)
        results['short_term'] = recent
        
        # 2. Buscar en memoria a largo plazo
        long_results = self.long_term.search(query)
        results['long_term'] = long_results[:limit]
        
        # 3. Buscar en memoria episódica
        ep_results = self.episodic.get_by_context(query, limit)
        results['episodic'] = ep_results
        
        return results
    
    def add_experience(self, input_text, output_text, context=None):
        """
        Añade una experiencia a la memoria episódica
        """
        self.episodic.add({
            'input': input_text,
            'output': output_text,
            'context': context or 'general'
        })
    
    def add_knowledge(self, key, value, metadata=None):
        """
        Añade conocimiento a la memoria a largo plazo
        """
        self.long_term.add(key, value, metadata)
    
    def add_recent(self, item, embedding=None):
        """
        Añade item a la memoria a corto plazo
        """
        self.short_term.add(item, embedding)
    
    def get_recent(self, n=5):
        return self.short_term.get_recent(n)
    
    def get_all_long_term(self):
        return self.long_term.get_all()
    
    def get_all_episodic(self):
        return self.episodic.get_all()
    
    def clear_all(self):
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
        print("🧹 Todas las memorias limpiadas")
    
    def __repr__(self):
        return f"MemoryRetrieval(short={self.short_term.size()}, long={self.long_term.size()}, ep={self.episodic.size()})"

def prueba_retrieval():
    print("🧠 PROBANDO SISTEMA DE RECUPERACIÓN DE MEMORIA")
    print("="*40)
    
    memory = MemoryRetrieval()
    
    # Añadir conocimiento
    memory.add_knowledge("nombre_proyecto", "Cerebro Zero", {"tipo": "proyecto"})
    memory.add_knowledge("lenguaje_principal", "Python", {"tipo": "programacion"})
    memory.add_knowledge("framework", "NumPy", {"tipo": "libreria"})
    
    # Añadir experiencias
    memory.add_experience("Hola", "Hola, ¿cómo estás?", "saludo")
    memory.add_experience("¿Qué hora es?", "Son las 11:00", "tiempo")
    memory.add_experience("¿Cómo funciona el cerebro?", "Usa redes neuronales", "educacion")
    
    # Añadir a corto plazo
    memory.add_recent("Última consulta: clima")
    memory.add_recent("Última consulta: noticias")
    
    print(f"📊 Estado: {memory}")
    
    # Buscar
    print("\n🔍 Buscando 'proyecto':")
    resultados = memory.remember("proyecto")
    for key, value in resultados.items():
        print(f"   {key}: {value}")
    
    print("\n🔍 Buscando 'saludo':")
    resultados = memory.remember("saludo")
    for key, value in resultados.items():
        print(f"   {key}: {value}")
    
    print("\n📋 Memoria a corto plazo:")
    print(f"   {memory.get_recent(3)}")
    
    print("\n✅ SISTEMA DE RECUPERACIÓN FUNCIONANDO!")

if __name__ == "__main__":
    prueba_retrieval()
