"""
Tests de Tool System 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from tools.tool_system_v3 import (
    Schema, ToolSpec, ToolResult, ToolRegistryV3,
    tool_calculadora, tool_contar_palabras, tool_mayusculas, tool_fallar,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def schema_str():
    return Schema("str", requerido=True, descripcion="texto")


@pytest.fixture
def registry():
    reg = ToolRegistryV3()
    reg.registrar(ToolSpec(
        nombre="calculadora",
        descripcion="Suma o multiplica",
        funcion=tool_calculadora,
        input_schema={
            "expresion": Schema("str", requerido=True),
            "precision": Schema("int", requerido=False, default=2, min=0, max=10),
        },
        output_tipo="float",
        capacidades=["matemáticas", "cálculo"],
    ))
    reg.registrar(ToolSpec(
        nombre="contar_palabras",
        descripcion="Cuenta palabras",
        funcion=tool_contar_palabras,
        input_schema={"texto": Schema("str", requerido=True)},
        output_tipo="int",
        capacidades=["texto"],
    ))
    reg.registrar(ToolSpec(
        nombre="mayusculas",
        descripcion="A mayúsculas",
        funcion=tool_mayusculas,
        input_schema={"texto": Schema("str", requerido=True)},
        output_tipo="str",
        capacidades=["texto", "transformación"],
    ))
    reg.registrar(ToolSpec(
        nombre="fallar",
        descripcion="Tool que falla",
        funcion=tool_fallar,
        capacidades=["test"],
    ))
    reg.registrar(ToolSpec(
        nombre="peligrosa",
        descripcion="Con permisos",
        funcion=tool_mayusculas,
        input_schema={"texto": Schema("str")},
        permisos=["admin"],
        capacidades=["test"],
    ))
    return reg


# ============================================
# TESTS DE SCHEMA
# ============================================

def test_schema_creation(schema_str):
    assert schema_str.tipo == "str"
    assert schema_str.requerido == True


def test_schema_validar_str_ok(schema_str):
    ok, err = schema_str.validar("hola")
    assert ok == True
    assert err is None


def test_schema_validar_str_falla(schema_str):
    ok, err = schema_str.validar(123)
    assert ok == False
    assert "str" in err


def test_schema_validar_none_requerido(schema_str):
    ok, err = schema_str.validar(None)
    assert ok == False


def test_schema_validar_none_no_requerido():
    s = Schema("str", requerido=False)
    ok, err = s.validar(None)
    assert ok == True


def test_schema_validar_int_min():
    s = Schema("int", min=5)
    ok, err = s.validar(3)
    assert ok == False
    assert "min" in err


def test_schema_validar_int_max():
    s = Schema("int", max=10)
    ok, err = s.validar(15)
    assert ok == False
    assert "max" in err


def test_schema_validar_int_rango_ok():
    s = Schema("int", min=0, max=10)
    ok, _ = s.validar(5)
    assert ok == True


def test_schema_validar_any():
    s = Schema("any")
    assert s.validar("x")[0] == True
    assert s.validar(42)[0] == True
    assert s.validar([1, 2])[0] == True


def test_schema_validar_list():
    s = Schema("list")
    assert s.validar([1, 2])[0] == True
    assert s.validar("no")[0] == False


def test_schema_validar_dict():
    s = Schema("dict")
    assert s.validar({"a": 1})[0] == True
    assert s.validar([1])[0] == False


def test_schema_validar_bool():
    s = Schema("bool")
    assert s.validar(True)[0] == True
    assert s.validar("yes")[0] == False


def test_schema_validar_float_acepta_int():
    s = Schema("float")
    assert s.validar(5)[0] == True
    assert s.validar(5.5)[0] == True


def test_schema_to_dict(schema_str):
    d = schema_str.to_dict()
    assert d["tipo"] == "str"
    assert d["requerido"] == True


# ============================================
# TESTS DE TOOL SPEC
# ============================================

def test_spec_creation():
    spec = ToolSpec(nombre="x", descripcion="y", funcion=lambda: None)
    assert spec.nombre == "x"
    assert spec.timeout == 5.0


def test_spec_to_dict():
    spec = ToolSpec(
        nombre="x", descripcion="y", funcion=lambda: None,
        capacidades=["a", "b"],
    )
    d = spec.to_dict()
    assert d["nombre"] == "x"
    assert d["capacidades"] == ["a", "b"]


def test_spec_validar_input_ok():
    spec = ToolSpec(
        nombre="x", descripcion="y", funcion=lambda: None,
        input_schema={"a": Schema("int", requerido=True)},
    )
    ok, err = spec.validar_input({"a": 5})
    assert ok == True


def test_spec_validar_input_falta_requerido():
    spec = ToolSpec(
        nombre="x", descripcion="y", funcion=lambda: None,
        input_schema={"a": Schema("int", requerido=True)},
    )
    ok, err = spec.validar_input({})
    assert ok == False
    assert "a" in err


def test_spec_validar_input_tipo_malo():
    spec = ToolSpec(
        nombre="x", descripcion="y", funcion=lambda: None,
        input_schema={"a": Schema("int", requerido=True)},
    )
    ok, err = spec.validar_input({"a": "no"})
    assert ok == False


def test_spec_aplicar_defaults():
    spec = ToolSpec(
        nombre="x", descripcion="y", funcion=lambda: None,
        input_schema={
            "a": Schema("int", requerido=True),
            "b": Schema("int", requerido=False, default=42),
        },
    )
    kwargs = spec.aplicar_defaults({"a": 1})
    assert kwargs["b"] == 42


# ============================================
# TESTS DE TOOL RESULT
# ============================================

def test_result_ok():
    r = ToolResult(ok=True, output=42, tool="x")
    assert r.ok == True
    assert r.output == 42


def test_result_error():
    r = ToolResult(ok=False, error="fallo", tool="x")
    assert r.ok == False
    assert r.error == "fallo"


def test_result_to_dict():
    r = ToolResult(ok=True, output="hola", tool="x")
    d = r.to_dict()
    assert d["ok"] == True
    assert d["tool"] == "x"


def test_result_repr_ok():
    r = ToolResult(ok=True, tool="x")
    assert "ok" in repr(r)


def test_result_repr_error():
    r = ToolResult(ok=False, tool="x", error="boom")
    assert "error" in repr(r)


# ============================================
# TESTS DE REGISTRY — REGISTRO
# ============================================

def test_registry_creation():
    reg = ToolRegistryV3()
    assert len(reg.listar()) == 0


def test_registry_registrar(registry):
    assert "calculadora" in registry.listar()
    assert "fallar" in registry.listar()


def test_registry_registrar_duplicado():
    reg = ToolRegistryV3()
    spec = ToolSpec(nombre="x", descripcion="y", funcion=lambda: None)
    reg.registrar(spec)
    with pytest.raises(ValueError):
        reg.registrar(spec)


def test_registry_desregistrar(registry):
    assert registry.desregistrar("calculadora") == True
    assert "calculadora" not in registry.listar()


def test_registry_desregistrar_no_existe(registry):
    assert registry.desregistrar("no_existe") == False


def test_registry_obtener(registry):
    spec = registry.obtener("calculadora")
    assert spec is not None
    assert spec.nombre == "calculadora"


def test_registry_obtener_no_existe(registry):
    assert registry.obtener("no_existe") is None


def test_registry_existe(registry):
    assert registry.existe("calculadora") == True
    assert registry.existe("no_existe") == False


def test_registry_listar_specs(registry):
    specs = registry.listar_specs()
    assert len(specs) == 5
    assert all("nombre" in s for s in specs)


def test_registry_repr(registry):
    assert "ToolRegistryV3" in repr(registry)


# ============================================
# TESTS DE BÚSQUEDA
# ============================================

def test_buscar_por_capacidad(registry):
    tools = registry.buscar_por_capacidad("texto")
    nombres = [t.nombre for t in tools]
    assert "contar_palabras" in nombres
    assert "mayusculas" in nombres


def test_buscar_por_capacidad_vacia(registry):
    tools = registry.buscar_por_capacidad("inexistente")
    assert tools == []


def test_buscar_por_capacidades(registry):
    """TODAS las capacidades"""
    tools = registry.buscar_por_capacidades(["texto", "transformación"])
    nombres = [t.nombre for t in tools]
    assert "mayusculas" in nombres
    assert "contar_palabras" not in nombres


def test_buscar_por_capacidades_multiples(registry):
    tools = registry.buscar_por_capacidades(["matemáticas", "cálculo"])
    nombres = [t.nombre for t in tools]
    assert "calculadora" in nombres


def test_buscar_texto(registry):
    tools = registry.buscar("palabras")
    assert len(tools) >= 1
    assert tools[0].nombre == "contar_palabras"


def test_buscar_texto_case_insensitive(registry):
    tools1 = registry.buscar("PALABRAS")
    tools2 = registry.buscar("palabras")
    assert len(tools1) == len(tools2)


# ============================================
# TESTS DE EJECUCIÓN
# ============================================

def test_ejecutar_ok(registry):
    r = registry.ejecutar("calculadora", {"expresion": "5+3"})
    assert r.ok == True
    assert r.output == 8.0


def test_ejecutar_con_default(registry):
    r = registry.ejecutar("calculadora", {"expresion": "1.23456+1"})
    assert r.ok == True
    # precision default = 2
    assert r.output == 2.23


def test_ejecutar_no_existe(registry):
    r = registry.ejecutar("no_existe")
    assert r.ok == False
    assert "no registrada" in r.error


def test_ejecutar_falta_requerido(registry):
    r = registry.ejecutar("calculadora", {})
    assert r.ok == False
    assert "requerido" in r.error


def test_ejecutar_tipo_malo(registry):
    r = registry.ejecutar("contar_palabras", {"texto": 123})
    assert r.ok == False


def test_ejecutar_min_max(registry):
    r = registry.ejecutar("calculadora",
                           {"expresion": "5+3", "precision": 99})
    assert r.ok == False
    assert "max" in r.error


def test_ejecutar_error_tool(registry):
    r = registry.ejecutar("fallar")
    assert r.ok == False
    assert "RuntimeError" in r.error


def test_ejecutar_contar(registry):
    r = registry.ejecutar("contar_palabras", {"texto": "uno dos tres"})
    assert r.output == 3


def test_ejecutar_mayusculas(registry):
    r = registry.ejecutar("mayusculas", {"texto": "hola"})
    assert r.output == "HOLA"


# ============================================
# TESTS DE PERMISOS
# ============================================

def test_ejecutar_sin_validador_pasa(registry):
    """Sin validador, los permisos no bloquean"""
    r = registry.ejecutar("peligrosa", {"texto": "hola"})
    assert r.ok == True


def test_ejecutar_con_validador_deniega(registry):
    def validador(permisos, tool):
        return False
    r = registry.ejecutar("peligrosa", {"texto": "hola"},
                           validador_permisos=validador)
    assert r.ok == False
    assert "Permisos" in r.error


def test_ejecutar_con_validador_permite(registry):
    def validador(permisos, tool):
        return True
    r = registry.ejecutar("peligrosa", {"texto": "hola"},
                           validador_permisos=validador)
    assert r.ok == True


def test_ejecutar_validador_error(registry):
    def validador(permisos, tool):
        raise RuntimeError("validador roto")
    r = registry.ejecutar("peligrosa", {"texto": "hola"},
                           validador_permisos=validador)
    assert r.ok == False
    assert "validador" in r.error.lower()


def test_ejecutar_tool_sin_permisos_no_llama_validador(registry):
    """Tools sin permisos no deben llamar al validador"""
    llamadas = []
    def validador(permisos, tool):
        llamadas.append(tool)
        return True
    registry.ejecutar("mayusculas", {"texto": "x"},
                       validador_permisos=validador)
    assert "mayusculas" not in llamadas


# ============================================
# TESTS DE STATS
# ============================================

def test_stats_vacio():
    reg = ToolRegistryV3()
    s = reg.stats()
    assert s["total"] == 0


def test_stats_con_ejecuciones(registry):
    registry.ejecutar("calculadora", {"expresion": "1+1"})
    registry.ejecutar("contar_palabras", {"texto": "a b c"})
    registry.ejecutar("fallar")
    s = registry.stats()
    assert s["total"] == 3
    assert s["exitos"] == 2
    assert s["tasa_exito"] == pytest.approx(2/3)


# ============================================
# TESTS DE PERSISTENCIA
# ============================================

def test_guardar_esquema(registry):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        registry.guardar_esquema(tmp)
        assert os.path.exists(tmp)
        import json
        with open(tmp) as f:
            data = json.load(f)
        assert "tools" in data
        assert len(data["tools"]) == 5
    finally:
        os.unlink(tmp)


def test_guardar_esquema_contiene_specs(registry):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        registry.guardar_esquema(tmp)
        import json
        with open(tmp) as f:
            data = json.load(f)
        
        nombres = [t["nombre"] for t in data["tools"]]
        assert "calculadora" in nombres
        assert "contar_palabras" in nombres
    finally:
        os.unlink(tmp)


# ============================================
# TESTS DE HISTORIAL
# ============================================

def test_historial_crece(registry):
    assert len(registry.historial) == 0
    registry.ejecutar("mayusculas", {"texto": "x"})
    registry.ejecutar("mayusculas", {"texto": "y"})
    assert len(registry.historial) == 2


def test_historial_incluye_fallos(registry):
    registry.ejecutar("fallar")
    assert len(registry.historial) == 1
    assert registry.historial[0].ok == False


# ============================================
# TESTS DE INTEGRACIÓN
# ============================================

def test_flujo_completo(registry):
    """Flujo típico: registrar → buscar → ejecutar"""
    tools = registry.buscar_por_capacidad("matemáticas")
    assert len(tools) == 1
    
    spec = tools[0]
    r = registry.ejecutar(spec.nombre, {"expresion": "10*5"})
    assert r.ok == True
    assert r.output == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
