"""
Tests de Working Memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from memory.working_memory import WorkingMemory, WorkingItem


# ============================================
# TESTS BÁSICOS
# ============================================

def test_wm_creation():
    wm = WorkingMemory(max_items=5)
    assert wm.size() == 0
    assert wm.max_items == 5


def test_wm_add():
    wm = WorkingMemory()
    wm.add("key1", "value1", prioridad=0.8)
    assert wm.size() == 1
    assert wm.get("key1") == "value1"


def test_wm_get_nonexistent():
    wm = WorkingMemory()
    assert wm.get("no_existe") is None


def test_wm_update():
    wm = WorkingMemory()
    wm.add("key", "value1")
    wm.add("key", "value2")
    assert wm.size() == 1
    assert wm.get("key") == "value2"


# ============================================
# TESTS DE CAPACIDAD
# ============================================

def test_wm_max_capacity():
    wm = WorkingMemory(max_items=3)
    wm.add("a", 1)
    wm.add("b", 2)
    wm.add("c", 3)
    wm.add("d", 4)
    # Debe haber expulsado uno
    assert wm.size() == 3


def test_wm_expulsa_menos_relevante():
    wm = WorkingMemory(max_items=3)
    wm.add("importante", "v1", prioridad=1.0)
    wm.add("medio", "v2", prioridad=0.5)
    wm.add("poco", "v3", prioridad=0.1)
    
    # Añadir uno nuevo → debe expulsar "poco"
    wm.add("nuevo", "v4", prioridad=0.9)
    
    assert "importante" in wm.keys()
    assert "nuevo" in wm.keys()
    assert "poco" not in wm.keys()


# ============================================
# TESTS DE TTL
# ============================================

def test_wm_ttl_expira():
    wm = WorkingMemory(max_items=5, default_ttl=0.5)
    wm.add("temp", "valor")
    assert wm.size() == 1
    time.sleep(0.7)
    wm.tick()
    assert wm.size() == 0


def test_wm_ttl_no_expira():
    wm = WorkingMemory(max_items=5, default_ttl=10.0)
    wm.add("permanente", "valor")
    time.sleep(0.2)
    wm.tick()
    assert wm.size() == 1


def test_wm_get_expirado_devuelve_none():
    wm = WorkingMemory(default_ttl=0.3)
    wm.add("temp", "valor")
    time.sleep(0.5)
    assert wm.get("temp") is None


# ============================================
# TESTS DE RELEVANCIA
# ============================================

def test_item_relevancia():
    item = WorkingItem(key="test", value="val", prioridad=0.8, ttl=10.0)
    rel = item.relevancia()
    assert 0.0 <= rel <= 1.0


def test_get_top_relevantes():
    wm = WorkingMemory(max_items=5, default_ttl=60.0)
    wm.add("a", 1, prioridad=0.3)
    wm.add("b", 2, prioridad=0.9)
    wm.add("c", 3, prioridad=0.5)
    
    top = wm.get_top_relevantes(k=2)
    assert len(top) == 2
    assert top[0].prioridad >= top[1].prioridad


def test_get_relevante_query():
    wm = WorkingMemory(default_ttl=60.0)
    wm.add("dataset_train", "train.txt")
    wm.add("dataset_test", "test.txt")
    wm.add("modelo", "transformer")
    
    matching = wm.get_relevante("dataset")
    assert len(matching) == 2


# ============================================
# TESTS DE ACCESO
# ============================================

def test_acceso_refuerza():
    wm = WorkingMemory(default_ttl=60.0)
    wm.add("key", "value")
    
    item = wm.items["key"]
    rel_inicial = item.relevancia()
    
    # Acceder varias veces
    for _ in range(5):
        wm.get("key")
    
    rel_final = wm.items["key"].relevancia()
    assert rel_final >= rel_inicial


def test_veces_accedido():
    wm = WorkingMemory(default_ttl=60.0)
    wm.add("key", "value")
    assert wm.items["key"].veces_accedido == 0
    
    wm.get("key")
    wm.get("key")
    wm.get("key")
    
    assert wm.items["key"].veces_accedido == 3


# ============================================
# TESTS DE UTILIDADES
# ============================================

def test_wm_clear():
    wm = WorkingMemory()
    wm.add("a", 1)
    wm.add("b", 2)
    wm.clear()
    assert wm.size() == 0


def test_wm_keys():
    wm = WorkingMemory()
    wm.add("a", 1)
    wm.add("b", 2)
    keys = wm.keys()
    assert "a" in keys
    assert "b" in keys


def test_wm_resumen():
    wm = WorkingMemory(default_ttl=60.0)
    wm.add("a", 1, prioridad=0.9)
    wm.add("b", 2, prioridad=0.5)
    
    resumen = wm.resumen()
    assert resumen['size'] == 2
    assert resumen['max_items'] == 10
    assert len(resumen['items']) == 2


def test_wm_is_full():
    wm = WorkingMemory(max_items=2)
    wm.add("a", 1)
    assert not wm.is_full()
    wm.add("b", 2)
    assert wm.is_full()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
