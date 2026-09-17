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
from memory.semantic import SemanticMemory


class MemoryRetrieval:
    def __init__(self):
        self.short_term = ShortTermMemory(max_size=100)
        self.long_term = LongTermMemory(filename="memoria_largo_plazo.json")
        self.episodic = EpisodicMemory(max_size=100)
        self.semantic = SemanticMemory(archivo="memoria_semantica.json", dim=128)
    
    def remember(self, query, limit=5):
        """
        Recupera información relevante de todas las memorias.
        """
        results = {
            'short_term': [],
            'long_term': [],
            'episodic': [],
            'semantic': [],
        }
        
        # 1. Corto plazo
        recent = self.short_term.get_recent(limit)
        results['short_term'] = recent
        
        # 2. Largo plazo (búsqueda por coincidencia exacta)
        long_results = self.long_term.search(query)
        results['long_term'] = long_results[:limit]
        
        # 3. Episódica (búsqueda por contexto)
        ep_results = self.episodic.get_by_context(query, limit)
        results['episodic'] = ep_results
        
        # 4. Semántica (búsqueda por similitud)
        sem_results = self.semantic.search(query, k=limit)
        results['semantic'] = sem_results
        
        return results
    
    def add_experience(self, input_text, output_text, context=None):
        """Añade una experiencia a la memoria episódica"""
        self.episodic.add({
            'input': input_text,
            'output': output_text,
            'context': context or 'general'
        })
        # También añadir a semántica para búsqueda por similitud
        self.semantic.add(
            f"{input_text} → {output_text}",
            {'tipo': 'experiencia', 'contexto': context or 'general'}
        )
    
    def add_knowledge(self, key, value, metadata=None):
        """Añade conocimiento a largo plazo"""
        self.long_term.add(key, value, metadata)
        # También añadir a semántica
        self.semantic.add(
            f"{key}: {value}",
            {'tipo': 'conocimiento', **(metadata or {})}
        )
    
    def add_recent(self, item, embedding=None):
        """Añade item a la memoria a corto plazo"""
        self.short_term.add(item, embedding)
    
    def get_recent(self, n=5):
        return self.short_term.get_recent(n)
    
    def get_all_long_term(self):
        return self.long_term.get_all()
    
    def get_all_episodic(self):
        return self.episodic.get_all()
    
    def get_all_semantic(self):
        return self.semantic.entradas
    
    def clear_all(self):
        self.short_term.clear()
        self.long_term.clear()
        self.episodic.clear()
        self.semantic.clear()
        print("🧹 Todas las memorias limpiadas")
    
    def __repr__(self):
        return (f"MemoryRetrieval("
                f"short={self.short_term.size()}, "
                f"long={self.long_term.size()}, "
                f"ep={self.episodic.size()}, "
                f"sem={self.semantic.size()})")


if __name__ == "__main__":
    print("🧪 PROBANDO MEMORY RETRIEVAL CON SEMÁNTICA")
    print("="*50)
    
    mem = MemoryRetrieval()
    
    # Añadir conocimiento
    mem.add_knowledge("proyecto", "Cerebro Zero", {"tipo": "proyecto"})
    mem.add_knowledge("lenguaje", "Python", {"tipo": "programacion"})
    
    # Añadir experiencias
    mem.add_experience("hola", "Hola, ¿qué tal?", context="saludo")
    mem.add_experience("adiós", "Hasta luego", context="despedida")
    
    print(f"\n📊 Estado: {mem}")
    
    # Buscar
    print("\n🔍 Búsquedas:")
    for q in ["hola", "proyecto", "Python", "despedida"]:
        print(f"\n   Query: '{q}'")
        r = mem.remember(q)
        for key, items in r.items():
            if items:
                print(f"      {key}: {len(items)} resultados")
                for item in items[:2]:
                    if isinstance(item, dict):
                        texto = item.get('texto') or item.get('output') or item.get('value') or str(item)
                        print(f"         → {texto[:60]}")
    
    print("\n✅ MEMORY RETRIEVAL CON SEMÁNTICA FUNCIONANDO")
