"""
Tests de Procedural Memory
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
from memory.procedural_memory import (
    ProceduralMemory, Procedimiento, Step,
)


# ============================================
# TESTS BÁSICOS
# ============================================

def test_pm_creation():
    pm = ProceduralMemory()
    assert pm.size() == 0


def test_pm_aprender():
    pm = ProceduralMemory()
    pm.aprender("sumar", r'suma|\+', [Step("calc", {}, "sumar")])
    assert pm.size() == 1
    assert "sumar" in pm.procedimientos


def test_pm_aprender_multiples():
    pm = ProceduralMemory()
    pm.aprender("sumar", r'suma', [Step("sumar")])
    pm.aprender("multiplicar", r'mult', [Step("mult")])
    assert pm.size() == 2


# ============================================
# TESTS DE BÚSQUEDA
# ============================================

def test_pm_buscar_simple():
    pm = ProceduralMemory()
    pm.aprender("sumar", r'suma|\+', [Step("calc")])
    
    proc = pm.buscar("¿Cuánto es 5 + 3?")
    assert proc is not None
    assert proc.nombre == "sumar"


def test_pm_buscar_no_encontrado():
    pm = ProceduralMemory()
    pm.aprender("sumar", r'suma', [Step("calc")])
    
    proc = pm.buscar("hola mundo")
    assert proc is None


def test_pm_buscar_por_tasa_exito():
    pm = ProceduralMemory()
    
    # Dos procedimientos con el mismo trigger
    p1 = pm.aprender("proc1", r'test', [Step("a")])
    p2 = pm.aprender("proc2", r'test', [Step("b")])
    
    # p1 tiene más éxito
    for _ in range(10):
        p1.registrar_exito()
    p2.registrar_exito()
    
    proc = pm.buscar("test")
    assert proc.nombre == "proc1"


# ============================================
# TESTS DE EJECUCIÓN
# ============================================

def test_pm_ejecutar_sin_ejecutores():
    pm = ProceduralMemory()
    pm.aprender("test", r'test', [Step("paso1"), Step("paso2")])
    
    proc = pm.procedimientos["test"]
    resultado = pm.ejecutar(proc, contexto={})
    
    assert resultado['exito']
    assert len(resultado['pasos_ejecutados']) == 2


def test_pm_ejecutar_con_ejecutores():
    pm = ProceduralMemory()
    pm.aprender("sumar", r'suma', [
        Step("suma", {"a": 5, "b": 3}),
    ])
    
    def suma(ctx, params):
        return {'resultado': params['a'] + params['b']}
    
    proc = pm.procedimientos["sumar"]
    resultado = pm.ejecutar(
        proc,
        contexto={},
        ejecutores={'suma': suma},
    )
    
    assert resultado['exito']
    assert resultado['contexto_final']['resultado'] == 8


def test_pm_ejecutar_error():
    pm = ProceduralMemory()
    pm.aprender("test", r'test', [Step("fallo")])
    
    def fallo(ctx, params):
        raise ValueError("Error forzado")
    
    proc = pm.procedimientos["test"]
    resultado = pm.ejecutar(
        proc,
        contexto={},
        ejecutores={'fallo': fallo},
    )
    
    assert not resultado['exito']
    assert "Error forzado" in resultado['error']


# ============================================
# TESTS DE ÉXITO/FALLO
# ============================================

def test_procedimiento_tasa_exito():
    proc = Procedimiento(nombre="test", pasos=[], trigger="test")
    assert proc.tasa_exito == 0.5  # default
    
    proc.registrar_exito()
    proc.registrar_exito()
    proc.registrar_exito()
    proc.registrar_fallo()
    
    assert proc.tasa_exito == 0.75  # 3/4


def test_aprender_de_exito():
    pm = ProceduralMemory()
    proc = pm.aprender_de_exito(
        entrada="5 + 3",
        contexto={},
        pasos=[Step("calc")],
    )
    assert proc.exitos == 1
    assert proc.tasa_exito == 1.0


# ============================================
# TESTS DE CAPACIDAD
# ============================================

def test_pm_max_capacidad():
    pm = ProceduralMemory(max_procedimientos=3)
    pm.aprender("p1", r'1', [Step("a")])
    pm.aprender("p2", r'2', [Step("b")])
    pm.aprender("p3", r'3', [Step("c")])
    
    # Los tres primeros usan 3
    pm.procedimientos["p1"].registrar_exito()
    pm.procedimientos["p2"].registrar_exito()
    # p3 no se usa
    
    # Añadir uno nuevo → expulsa p3
    pm.aprender("p4", r'4', [Step("d")])
    
    assert pm.size() == 3
    assert "p4" in pm.procedimientos
    assert "p3" not in pm.procedimientos


# ============================================
# TESTS DE UTILIDADES
# ============================================

def test_pm_olvidar():
    pm = ProceduralMemory()
    pm.aprender("test", r'test', [Step("a")])
    assert pm.olvidar("test") == True
    assert pm.size() == 0
    assert pm.olvidar("no_existe") == False


def test_pm_clear():
    pm = ProceduralMemory()
    pm.aprender("p1", r'1', [Step("a")])
    pm.aprender("p2", r'2', [Step("b")])
    pm.clear()
    assert pm.size() == 0


def test_pm_listar():
    pm = ProceduralMemory()
    p1 = pm.aprender("p1", r'1', [Step("a")])
    p2 = pm.aprender("p2", r'2', [Step("b")])
    p1.registrar_exito()
    p1.registrar_exito()
    p2.registrar_fallo()
    
    lista = pm.listar()
    assert len(lista) == 2
    # p1 debe estar primero (más éxito)
    assert lista[0]['nombre'] == "p1"


def test_pm_resumen():
    pm = ProceduralMemory()
    pm.aprender("test", r'test', [Step("a")])
    resumen = pm.resumen()
    assert resumen['size'] == 1
    assert len(resumen['procedimientos']) == 1


def test_step_to_dict():
    step = Step("calcular", {"op": "+"}, "Sumar")
    d = step.to_dict()
    assert d['accion'] == "calcular"
    assert d['parametros'] == {"op": "+"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
