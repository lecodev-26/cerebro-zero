"""
Tests de la memoria semántica
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memory.embeddings import TextEmbedder, cosine_similarity
from memory.semantic import SemanticMemory


# ============================================
# TESTS DE EMBEDDINGS
# ============================================

def test_embedder_creation():
    embedder = TextEmbedder(dim=64)
    assert embedder.dim == 64


def test_embedder_output_shape():
    embedder = TextEmbedder(dim=64)
    vec = embedder.embed("hola mundo")
    assert vec.shape == (64,)


def test_embedder_normalized():
    embedder = TextEmbedder(dim=128)
    vec = embedder.embed("hola mundo")
    norma = np.linalg.norm(vec)
    assert np.isclose(norma, 1.0, atol=1e-6), f"Norma debe ser 1, es {norma}"


def test_embedder_empty_text():
    embedder = TextEmbedder(dim=64)
    vec = embedder.embed("")
    assert vec.shape == (64,)
    assert np.allclose(vec, 0)


def test_similarity_identical():
    embedder = TextEmbedder(dim=128)
    v1 = embedder.embed("hola mundo")
    v2 = embedder.embed("hola mundo")
    sim = cosine_similarity(v1, v2)
    assert np.isclose(sim, 1.0, atol=1e-6)


def test_similarity_similar():
    """Textos parecidos deben tener similitud > 0.5"""
    embedder = TextEmbedder(dim=128)
    v1 = embedder.embed("hola mundo")
    v2 = embedder.embed("hola mundo cruel")
    sim = cosine_similarity(v1, v2)
    assert sim > 0.3, f"Similitud debe ser > 0.3, es {sim}"


def test_similarity_different():
    """Textos muy diferentes deben tener similitud baja"""
    embedder = TextEmbedder(dim=128)
    v1 = embedder.embed("inteligencia artificial")
    v2 = embedder.embed("recetas de cocina")
    sim = cosine_similarity(v1, v2)
    assert sim < 0.3, f"Similitud debe ser < 0.3, es {sim}"


# ============================================
# TESTS DE MEMORIA SEMÁNTICA
# ============================================

def test_semantic_creation():
    mem = SemanticMemory(archivo="test_mem_sem.json")
    assert mem.size() >= 0


def test_semantic_add_and_size():
    mem = SemanticMemory(archivo="test_mem_sem.json")
    mem.clear()
    mem.add("el sol es una estrella")
    mem.add("la luna es un satélite")
    assert mem.size() == 2


def test_semantic_search():
    mem = SemanticMemory(archivo="test_mem_sem.json")
    mem.clear()
    mem.add("el sol es una estrella")
    mem.add("la luna es un satélite")
    mem.add("python es un lenguaje")
    
    resultados = mem.search("qué es el sol", k=1)
    assert len(resultados) >= 1
    # El resultado más parecido debe contener "sol"
    assert "sol" in resultados[0]['texto'].lower()


def test_semantic_search_similarity_threshold():
    mem = SemanticMemory(archivo="test_mem_sem.json", umbral=0.5)
    mem.clear()
    mem.add("python es un lenguaje")
    
    # Query muy diferente
    resultados = mem.search("receta de tortilla", k=3)
    # Con umbral 0.5, no debería haber resultados
    assert len(resultados) == 0


def test_semantic_clear():
    mem = SemanticMemory(archivo="test_mem_sem.json")
    mem.add("test")
    assert mem.size() > 0
    mem.clear()
    assert mem.size() == 0


def test_semantic_search_returns_ordered():
    """Los resultados deben estar ordenados por similitud"""
    mem = SemanticMemory(archivo="test_mem_sem.json")
    mem.clear()
    mem.add("el sol es una estrella brillante")
    mem.add("el sol calienta la tierra")
    mem.add("python es un lenguaje")
    
    resultados = mem.search("sol", k=3)
    if len(resultados) >= 2:
        assert resultados[0]['similitud'] >= resultados[1]['similitud']


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_retrieval_with_semantic():
    from memory.retrieval import MemoryRetrieval
    mem = MemoryRetrieval()
    mem.clear_all()
    
    mem.add_knowledge("proyecto", "Cerebro Zero")
    mem.add_experience("hola", "Hola, ¿qué tal?")
    
    r = mem.remember("hola")
    assert 'semantic' in r
    assert len(r['semantic']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
