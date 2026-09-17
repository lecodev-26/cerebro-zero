"""
Tests de World Model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from memory.world_model import WorldModel, Cambio


@pytest.fixture
def wm():
    tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    w = WorldModel(archivo=tmpfile)
    w.clear()
    yield w
    try:
        os.unlink(tmpfile)
    except:
        pass


# ============================================
# TESTS BÁSICOS
# ============================================

def test_wm_creation(wm):
    assert wm.get("no.existe") is None


def test_wm_set_get(wm):
    wm.set("a", 1)
    assert wm.get("a") == 1


def test_wm_set_nested(wm):
    wm.set("user.nombre", "Manuel")
    assert wm.get("user.nombre") == "Manuel"


def test_wm_set_deeply_nested(wm):
    wm.set("a.b.c.d", "profundo")
    assert wm.get("a.b.c.d") == "profundo"


def test_wm_get_default(wm):
    assert wm.get("no.existe", 42) == 42


# ============================================
# TESTS DE UPDATE
# ============================================

def test_wm_update(wm):
    wm.set("key", "v1")
    wm.update("key", "v2")
    assert wm.get("key") == "v2"


def test_wm_update_nested(wm):
    wm.set("user.nombre", "Manuel")
    wm.update("user.nombre", "Manu")
    assert wm.get("user.nombre") == "Manu"


# ============================================
# TESTS DE DELETE
# ============================================

def test_wm_delete(wm):
    wm.set("key", "value")
    assert wm.delete("key") == True
    assert wm.get("key") is None


def test_wm_delete_nonexistent(wm):
    assert wm.delete("no.existe") == False


def test_wm_delete_nested(wm):
    wm.set("user.nombre", "Manuel")
    wm.delete("user.nombre")
    assert wm.get("user.nombre") is None


# ============================================
# TESTS DE EXISTE
# ============================================

def test_wm_existe(wm):
    wm.set("key", "value")
    assert wm.existe("key") == True
    assert wm.existe("no.existe") == False


# ============================================
# TESTS DE INCREMENTAR
# ============================================

def test_wm_incrementar(wm):
    wm.set("contador", 0)
    wm.incrementar("contador")
    assert wm.get("contador") == 1


def test_wm_incrementar_delta(wm):
    wm.set("contador", 10)
    wm.incrementar("contador", 5)
    assert wm.get("contador") == 15


def test_wm_incrementar_sin_inicial(wm):
    wm.incrementar("nuevo", 3)
    assert wm.get("nuevo") == 3


# ============================================
# TESTS DE APPEND
# ============================================

def test_wm_append(wm):
    wm.set("lista", [])
    wm.append("lista", "a")
    wm.append("lista", "b")
    assert wm.get("lista") == ["a", "b"]


def test_wm_append_sin_inicial(wm):
    wm.append("nueva_lista", "x")
    assert wm.get("nueva_lista") == ["x"]


# ============================================
# TESTS DE HISTORIAL
# ============================================

def test_wm_historial(wm):
    wm.set("a", 1)
    wm.set("b", 2)
    historial = wm.get_historial()
    assert len(historial) >= 2


def test_wm_historial_contiene_cambios(wm):
    wm.set("key", "v1")
    wm.set("key", "v2")
    historial = wm.get_historial(5)
    # El último cambio debe tener v1 → v2
    ultimo = historial[-1]
    assert "v2" in ultimo['valor_nuevo']


# ============================================
# TESTS DE SUSCRIPCIONES
# ============================================

def test_wm_suscribir(wm):
    cambios = []
    wm.suscribir("key", lambda r, v, n: cambios.append((r, v, n)))
    
    wm.set("key", "nuevo")
    assert len(cambios) == 1
    assert cambios[0] == ("key", None, "nuevo")


def test_wm_suscribir_multiple(wm):
    cambios = []
    wm.suscribir("a", lambda r, v, n: cambios.append("a"))
    wm.suscribir("b", lambda r, v, n: cambios.append("b"))
    
    wm.set("a", 1)
    wm.set("b", 2)
    assert len(cambios) == 2


# ============================================
# TESTS DE UTILIDADES
# ============================================

def test_wm_keys(wm):
    wm.set("a", 1)
    wm.set("b", 2)
    keys = wm.keys()
    assert "a" in keys
    assert "b" in keys


def test_wm_clear(wm):
    wm.set("a", 1)
    wm.set("b", 2)
    wm.clear()
    assert wm.keys() == []


def test_wm_to_dict(wm):
    wm.set("user.nombre", "Manuel")
    d = wm.to_dict()
    assert d["user"]["nombre"] == "Manuel"


def test_wm_resumen(wm):
    wm.set("key", "value")
    resumen = wm.resumen()
    assert "key" in resumen['keys']


# ============================================
# TESTS DE PERSISTENCIA
# ============================================

def test_wm_persistencia(wm):
    wm.set("persistente", "valor")
    
    # Crear nueva instancia con el mismo archivo
    wm2 = WorldModel(archivo=wm.archivo)
    assert wm2.get("persistente") == "valor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================
# TESTS DE ENDURECIMIENTO (FASE 3.3 extra)
# ============================================

def test_wm_notifica_rutas_padre(wm):
    """Suscribirse a 'user' debe recibir cambios en 'user.nombre'"""
    cambios = []
    wm.suscribir("user", lambda r, v, n: cambios.append((r, v, n)))
    
    wm.set("user.nombre", "Manuel")
    assert len(cambios) == 1
    assert cambios[0][0] == "user.nombre"
    assert cambios[0][2] == "Manuel"


def test_wm_notifica_padre_y_hijo(wm):
    """Suscribirse a ambos debe recibir ambos"""
    cambios_padre = []
    cambios_hijo = []
    
    wm.suscribir("user", lambda r, v, n: cambios_padre.append(r))
    wm.suscribir("user.nombre", lambda r, v, n: cambios_hijo.append(r))
    
    wm.set("user.nombre", "Manuel")
    
    assert len(cambios_padre) == 1
    assert len(cambios_hijo) == 1


def test_wm_historial_limite_200(wm):
    """El historial no debe superar 200 cambios"""
    for i in range(250):
        wm.set(f"key_{i}", i)
    
    historial = wm.get_historial(1000)  # pedir más de los que hay
    assert len(historial) <= 200


def test_wm_incrementar_sobre_string_no_rompe(wm):
    """Incrementar sobre un string debe fallar limpiamente (devolver None o no cambiar)"""
    wm.set("texto", "hola")
    try:
        wm.incrementar("texto", 1)
        # Si no lanza excepción, al menos que no haya corrompido el estado
        valor = wm.get("texto")
        # El comportamiento aceptable: lanza excepción capturada o deja el valor intacto
        assert valor in ("hola", None) or isinstance(valor, (int, float))
    except (TypeError, ValueError):
        # También es aceptable que lance una excepción de tipo
        pass


def test_wm_set_dict_anidado(wm):
    """Set con dict en valor debe guardarse correctamente"""
    wm.set("config.debug", {"level": 3, "verbose": True})
    valor = wm.get("config.debug")
    assert valor == {"level": 3, "verbose": True}


def test_wm_overwrite_dict_con_valor(wm):
    """Sobrescribir un dict con un valor simple"""
    wm.set("config.debug", {"level": 3})
    wm.set("config.debug", "simple")
    assert wm.get("config.debug") == "simple"


def test_wm_set_sobrescribe_dict_con_dict(wm):
    """Sobrescribir ruta que era dict con otro dict"""
    wm.set("a.b.c", 1)
    wm.set("a.b", {"nuevo": 2})
    # Al sobrescribir "a.b" con dict, "a.b.c" desaparece
    assert wm.get("a.b.c") is None
    assert wm.get("a.b") == {"nuevo": 2}


def test_wm_suscribir_callback_error_no_rompe(wm):
    """Si un callback lanza error, no debe romper el set"""
    def callback_malo(r, v, n):
        raise ValueError("callback roto")
    
    wm.suscribir("key", callback_malo)
    # Esto no debe lanzar excepción hacia fuera
    resultado = wm.set("key", "valor")
    assert resultado == True
    assert wm.get("key") == "valor"


def test_wm_historial_tras_clear(wm):
    """Tras clear, el historial debe vaciarse"""
    wm.set("a", 1)
    wm.set("b", 2)
    assert len(wm.get_historial(10)) >= 2
    
    wm.clear()
    # Tras clear, el historial está vacío
    assert wm.get_historial(10) == []


def test_wm_resumen_completo(wm):
    """resumen() debe devolver estructura coherente"""
    wm.set("user.nombre", "Manuel")
    wm.set("user.edad", 25)
    wm.set("proyecto.status", "dev")
    
    resumen = wm.resumen()
    assert "keys" in resumen
    assert "total_claves" in resumen
    assert "archivo" in resumen
    assert "estado" in resumen
    assert "user" in resumen["keys"]
    assert "proyecto" in resumen["keys"]
