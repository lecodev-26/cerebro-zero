"""
Memoria a largo plazo (Long-Term Memory)
"""

import numpy as np
import pickle
import json
import os
from datetime import datetime

class LongTermMemory:
    def __init__(self, filename="memoria_largo_plazo.json", max_size=1000):
        self.filename = filename
        self.max_size = max_size
        self.memory = {}
        self.load()
    
    def add(self, key, value, metadata=None):
        """
        Añade información a la memoria
        """
        if len(self.memory) >= self.max_size:
            # Eliminar el más antiguo
            oldest = min(self.memory.keys(), key=lambda k: self.memory[k].get('timestamp', 0))
            del self.memory[oldest]
        
        self.memory[key] = {
            'value': value,
            'metadata': metadata or {},
            'timestamp': datetime.now().timestamp()
        }
        self.save()
    
    def get(self, key):
        """
        Obtiene información de la memoria
        """
        if key in self.memory:
            self.memory[key]['metadata']['last_access'] = datetime.now().timestamp()
            self.save()
            return self.memory[key]['value']
        return None
    
    def delete(self, key):
        """
        Elimina información de la memoria
        """
        if key in self.memory:
            del self.memory[key]
            self.save()
            return True
        return False
    
    def search(self, query):
        """
        Busca información por coincidencia parcial
        """
        results = []
        for key, data in self.memory.items():
            if query.lower() in key.lower():
                results.append({
                    'key': key,
                    'value': data['value'],
                    'metadata': data.get('metadata', {})
                })
        return results
    
    def get_all(self):
        """
        Obtiene toda la memoria
        """
        return self.memory
    
    def size(self):
        return len(self.memory)
    
    def save(self):
        """
        Guarda la memoria en disco
        """
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except:
            print(f"⚠️ No se pudo guardar {self.filename}")
    
    def load(self):
        """
        Carga la memoria desde disco
        """
        try:
            with open(self.filename, 'r') as f:
                self.memory = json.load(f)
        except:
            self.memory = {}
    
    def clear(self):
        """
        Limpia la memoria
        """
        self.memory = {}
        self.save()
    
    def __repr__(self):
        return f"LongTermMemory(size={len(self.memory)}, max={self.max_size})"

def prueba_long_term():
    print("🧠 PROBANDO MEMORIA A LARGO PLAZO")
    print("="*30)
    
    memory = LongTermMemory(filename="test_memory.json", max_size=10)
    
    # Añadir información
    memory.add("nombre", "Manuel", {"tipo": "personal"})
    memory.add("proyecto", "Cerebro Zero", {"tipo": "trabajo"})
    memory.add("lenguaje", "Python", {"tipo": "programación"})
    
    print(f"📋 Memoria: {memory.get_all()}")
    print(f"\n🔍 Buscar 'proyecto': {memory.search('proyecto')}")
    print(f"\n🔍 Buscar 'nombre': {memory.search('nombre')}")
    
    print(f"\n📝 Obtener 'proyecto': {memory.get('proyecto')}")
    
    memory.delete("nombre")
    print(f"\n🗑️ Eliminado 'nombre'")
    print(f"📋 Memoria: {memory.get_all()}")
    
    memory.clear()
    print(f"🧹 Limpiado: {memory.size()}")
    
    print("\n✅ MEMORIA A LARGO PLAZO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_long_term()
