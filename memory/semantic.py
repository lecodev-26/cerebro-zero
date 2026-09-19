"""
Memoria semántica con búsqueda vectorial
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from datetime import datetime
from memory.embeddings import TextEmbedder, cosine_similarity


class SemanticMemory:
    """
    Memoria semántica.
    
    Guarda conocimiento como (texto, embedding) y recupera por similitud coseno.
    
    Uso:
        mem = SemanticMemory()
        mem.add("el sol es una estrella", {"tipo": "astronomia"})
        resultados = mem.search("qué es el sol", k=3)
    """
    
    def __init__(self, archivo="semantic_memory.json", dim=128, umbral=0.3):
        self.archivo = archivo
        self.dim = dim
        self.umbral = umbral  # Similitud mínima para devolver
        self.embedder = TextEmbedder(dim=dim)
        self.entradas = []  # Lista de dicts con 'texto', 'metadata', 'embedding'
        self.cargar()
    
    def add(self, texto: str, metadata: dict = None):
        """Añade una entrada a la memoria semántica"""
        if not texto or not texto.strip():
            return
        
        embedding = self.embedder.embed(texto)
        
        entrada = {
            'texto': texto,
            'metadata': metadata or {},
            'embedding': embedding.tolist(),
            'fecha': datetime.now().isoformat(),
        }
        
        self.entradas.append(entrada)
        self.guardar()
    
    def search(self, query: str, k: int = 3, umbral: float = None):
        """
        Busca las k entradas más similares a la query.
        
        Args:
            query: texto de búsqueda
            k: número máximo de resultados
            umbral: similitud mínima (por defecto self.umbral)
        
        Returns:
            Lista de dicts con 'texto', 'metadata', 'similitud'
        """
        if not self.entradas:
            return []
        
        if umbral is None:
            umbral = self.umbral
        
        query_vec = self.embedder.embed(query)
        
        # Calcular similitud con todas las entradas
        resultados = []
        for entrada in self.entradas:
            vec = np.array(entrada['embedding'])
            sim = cosine_similarity(query_vec, vec)
            if sim >= umbral:
                resultados.append({
                    'texto': entrada['texto'],
                    'metadata': entrada['metadata'],
                    'similitud': sim,
                })
        
        # Ordenar por similitud descendente
        resultados.sort(key=lambda x: x['similitud'], reverse=True)
        
        return resultados[:k]
    
    def size(self) -> int:
        return len(self.entradas)
    
    def clear(self):
        self.entradas = []
        self.guardar()
    
    def guardar(self):
        """Guarda en disco"""
        path = os.path.join("../modelos_guardados", self.archivo)
        try:
            with open(path, 'w') as f:
                json.dump(self.entradas, f)
        except Exception as e:
            # Si falla, intentar guardar en directorio actual
            try:
                with open(self.archivo, 'w') as f:
                    json.dump(self.entradas, f)
            except Exception:
                pass
    
    def cargar(self):
        """Carga desde disco"""
        path = os.path.join("../modelos_guardados", self.archivo)
        try:
            with open(path, 'r') as f:
                self.entradas = json.load(f)
        except Exception:
            try:
                with open(self.archivo, 'r') as f:
                    self.entradas = json.load(f)
            except Exception:
                self.entradas = []
    
    def __repr__(self):
        return f"SemanticMemory({len(self.entradas)} entradas, dim={self.dim})"


if __name__ == "__main__":
    print("🧪 PROBANDO MEMORIA SEMÁNTICA")
    print("="*50)
    
    mem = SemanticMemory(archivo="test_semantic.json", dim=128)
    mem.clear()
    
    # Añadir conocimiento
    print("\n📚 Añadiendo conocimiento:")
    conocimientos = [
        ("El sol es una estrella que brilla en el cielo", {"tipo": "astronomia"}),
        ("La luna es el satélite natural de la Tierra", {"tipo": "astronomia"}),
        ("Python es un lenguaje de programación", {"tipo": "programacion"}),
        ("NumPy es una librería de cálculo numérico", {"tipo": "programacion"}),
        ("El cerebro procesa información y aprende", {"tipo": "biologia"}),
    ]
    for texto, meta in conocimientos:
        mem.add(texto, meta)
        print(f"   ✅ {texto[:50]}...")
    
    print(f"\n📊 Entradas: {mem.size()}")
    
    # Buscar
    print("\n🔍 Búsquedas:")
    queries = [
        "qué es el sol",
        "lenguaje de programación",
        "aprende información",
        "satélite de la Tierra",
    ]
    for q in queries:
        print(f"\n   Query: '{q}'")
        resultados = mem.search(q, k=2)
        for r in resultados:
            print(f"      → '{r['texto'][:50]}...' (sim={r['similitud']:.4f})")
    
    print("\n✅ MEMORIA SEMÁNTICA FUNCIONANDO")
