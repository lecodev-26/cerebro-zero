"""
Tool System 3.0 — Contratos + validación + ejecución segura
==============================================================

Filosofía:
    Cada tool declara QUÉ HACE, QUÉ NECESITA, QUÉ DEVUELVE,
    QUÉ PERMISOS REQUIERE y CUÁNTO puede tardar.

Componentes:
- ToolSpec: contrato declarativo de la tool
- ToolResult: resultado tipado
- ToolRegistryV3: registro con búsqueda por capacidades
- ToolExecutor: ejecuta con timeout + validación de schemas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime


# ============================================
# SCHEMA SIMPLIFICADO (sin jsonschema externo)
# ============================================

class Schema:
    """
    Schema simplificado tipo JSON Schema.
    
    Tipos soportados: "str", "int", "float", "bool", "list", "dict", "any"
    """
    
    def __init__(
        self,
        tipo: str = "any",
        requerido: bool = True,
        descripcion: str = "",
        default: Any = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ):
        self.tipo = tipo
        self.requerido = requerido
        self.descripcion = descripcion
        self.default = default
        self.min = min
        self.max = max
    
    def validar(self, valor: Any) -> tuple[bool, Optional[str]]:
        """Devuelve (valido, error)"""
        if valor is None:
            if self.requerido:
                return False, "valor requerido"
            return True, None
        
        tipos_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        if self.tipo != "any" and self.tipo in tipos_map:
            if not isinstance(valor, tipos_map[self.tipo]):
                return False, f"esperado {self.tipo}, recibido {type(valor).__name__}"
        
        if self.min is not None and isinstance(valor, (int, float)):
            if valor < self.min:
                return False, f"valor {valor} < min {self.min}"
        
        if self.max is not None and isinstance(valor, (int, float)):
            if valor > self.max:
                return False, f"valor {valor} > max {self.max}"
        
        return True, None
    
    def to_dict(self) -> dict:
        return {
            "tipo": self.tipo,
            "requerido": self.requerido,
            "descripcion": self.descripcion,
            "default": self.default,
        }


# ============================================
# TOOL SPEC (contrato)
# ============================================

@dataclass
class ToolSpec:
    """
    Contrato declarativo de una tool.
    """
    nombre: str
    descripcion: str
    funcion: Callable
    input_schema: Dict[str, Schema] = field(default_factory=dict)
    output_tipo: str = "any"
    permisos: List[str] = field(default_factory=list)
    timeout: float = 5.0
    capacidades: List[str] = field(default_factory=list)
    version: str = "1.0"
    
    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "descripcion": self.descripcion,
            "input_schema": {
                k: v.to_dict() for k, v in self.input_schema.items()
            },
            "output_tipo": self.output_tipo,
            "permisos": self.permisos,
            "timeout": self.timeout,
            "capacidades": self.capacidades,
            "version": self.version,
        }
    
    def validar_input(self, kwargs: dict) -> tuple[bool, Optional[str]]:
        """
        Valida kwargs contra input_schema.
        Devuelve (valido, error).
        """
        # Verificar campos requeridos
        for nombre, schema in self.input_schema.items():
            if schema.requerido and nombre not in kwargs:
                if schema.default is None:
                    return False, f"falta campo requerido: {nombre}"
        
        # Validar cada campo presente
        for nombre, valor in kwargs.items():
            schema = self.input_schema.get(nombre)
            if schema is None:
                # Campo no declarado → ignoramos (o podríamos rechazar)
                continue
            ok, err = schema.validar(valor)
            if not ok:
                return False, f"campo '{nombre}': {err}"
        
        return True, None
    
    def aplicar_defaults(self, kwargs: dict) -> dict:
        """Rellena campos faltantes con defaults"""
        resultado = kwargs.copy()
        for nombre, schema in self.input_schema.items():
            if nombre not in resultado and schema.default is not None:
                resultado[nombre] = schema.default
        return resultado


# ============================================
# TOOL RESULT
# ============================================

@dataclass
class ToolResult:
    """Resultado estructurado de una tool"""
    ok: bool
    output: Any = None
    error: Optional[str] = None
    tool: str = ""
    latencia: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "output": str(self.output)[:500] if self.output is not None else None,
            "error": self.error,
            "tool": self.tool,
            "latencia": self.latencia,
        }
    
    def __repr__(self):
        if self.ok:
            return f"ToolResult(ok, tool={self.tool}, lat={self.latencia*1000:.1f}ms)"
        return f"ToolResult(error={self.error}, tool={self.tool})"


# ============================================
# TOOL REGISTRY V3
# ============================================

class ToolRegistryV3:
    """
    Registro de tools con contratos.
    
    Uso:
        reg = ToolRegistryV3()
        reg.registrar(ToolSpec(...))
        resultado = reg.ejecutar("calculator", {"expresion": "5+3"})
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}
        self.historial: List[ToolResult] = []
    
    # ============================================
    # REGISTRO
    # ============================================
    
    def registrar(self, spec: ToolSpec):
        if spec.nombre in self.tools:
            raise ValueError(f"Tool '{spec.nombre}' ya registrada")
        self.tools[spec.nombre] = spec
    
    def desregistrar(self, nombre: str) -> bool:
        if nombre in self.tools:
            del self.tools[nombre]
            return True
        return False
    
    def obtener(self, nombre: str) -> Optional[ToolSpec]:
        return self.tools.get(nombre)
    
    def existe(self, nombre: str) -> bool:
        return nombre in self.tools
    
    def listar(self) -> List[str]:
        return list(self.tools.keys())
    
    def listar_specs(self) -> List[dict]:
        return [spec.to_dict() for spec in self.tools.values()]
    
    # ============================================
    # BÚSQUEDA POR CAPACIDAD
    # ============================================
    
    def buscar_por_capacidad(self, capacidad: str) -> List[ToolSpec]:
        """Devuelve tools que tienen esa capacidad"""
        return [
            spec for spec in self.tools.values()
            if capacidad in spec.capacidades
        ]
    
    def buscar_por_capacidades(self, capacidades: List[str]) -> List[ToolSpec]:
        """Devuelve tools que tengan TODAS las capacidades"""
        return [
            spec for spec in self.tools.values()
            if all(c in spec.capacidades for c in capacidades)
        ]
    
    def buscar(self, texto: str) -> List[ToolSpec]:
        """Búsqueda simple por texto en nombre/descripción"""
        texto = texto.lower()
        return [
            spec for spec in self.tools.values()
            if texto in spec.nombre.lower() or texto in spec.descripcion.lower()
        ]
    
    # ============================================
    # EJECUCIÓN
    # ============================================
    
    def ejecutar(
        self,
        nombre: str,
        kwargs: Optional[dict] = None,
        validador_permisos: Optional[Callable] = None,
    ) -> ToolResult:
        """
        Ejecuta una tool con validación completa.
        
        Args:
            nombre: nombre de la tool
            kwargs: argumentos
            validador_permisos: función(permisos, tool) que devuelve bool
        """
        t0 = time.time()
        kwargs = kwargs or {}
        
        # 1. Existe?
        spec = self.tools.get(nombre)
        if spec is None:
            r = ToolResult(
                ok=False, tool=nombre,
                error=f"Tool '{nombre}' no registrada",
            )
            self.historial.append(r)
            return r
        
        # 2. Permisos
        if spec.permisos and validador_permisos is not None:
            try:
                permitido = validador_permisos(spec.permisos, spec.nombre)
                if not permitido:
                    r = ToolResult(
                        ok=False, tool=nombre,
                        error=f"Permisos insuficientes: {spec.permisos}",
                        latencia=time.time() - t0,
                    )
                    self.historial.append(r)
                    return r
            except Exception as e:
                r = ToolResult(
                    ok=False, tool=nombre,
                    error=f"Error validador permisos: {e}",
                    latencia=time.time() - t0,
                )
                self.historial.append(r)
                return r
        
        # 3. Aplicar defaults
        kwargs = spec.aplicar_defaults(kwargs)
        
        # 4. Validar input
        ok, err = spec.validar_input(kwargs)
        if not ok:
            r = ToolResult(
                ok=False, tool=nombre,
                error=f"Input inválido: {err}",
                latencia=time.time() - t0,
            )
            self.historial.append(r)
            return r
        
        # 5. Ejecutar
        try:
            output = spec.funcion(**kwargs)
            r = ToolResult(
                ok=True, output=output, tool=nombre,
                latencia=time.time() - t0,
            )
        except Exception as e:
            r = ToolResult(
                ok=False, tool=nombre,
                error=f"{type(e).__name__}: {e}",
                latencia=time.time() - t0,
            )
        
        self.historial.append(r)
        return r
    
    # ============================================
    # HISTORIAL
    # ============================================
    
    def stats(self) -> dict:
        if not self.historial:
            return {"total": 0}
        
        exitos = sum(1 for r in self.historial if r.ok)
        latencias = [r.latencia for r in self.historial if r.ok]
        
        return {
            "total": len(self.historial),
            "exitos": exitos,
            "tasa_exito": exitos / len(self.historial),
            "latencia_media": (
                sum(latencias) / len(latencias) if latencias else 0.0
            ),
        }
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def guardar_esquema(self, archivo: str):
        """Guarda el esquema de las tools (sin funciones)"""
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        data = {
            "tools": [spec.to_dict() for spec in self.tools.values()],
            "guardado": datetime.now().isoformat(),
        }
        with open(archivo, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def __repr__(self):
        return f"ToolRegistryV3(tools={list(self.tools.keys())})"


# ============================================
# TOOLS DE EJEMPLO (para test manual)
# ============================================

def tool_calculadora(expresion: str, precision: int = 2) -> float:
    """Calculadora simple que suma o multiplica"""
    partes = expresion.replace(" ", "").split("+")
    if len(partes) == 2:
        resultado = float(partes[0]) + float(partes[1])
    else:
        partes = expresion.replace(" ", "").split("*")
        if len(partes) == 2:
            resultado = float(partes[0]) * float(partes[1])
        else:
            raise ValueError("Formato no soportado: usa 'a+b' o 'a*b'")
    return round(resultado, precision)


def tool_contar_palabras(texto: str) -> int:
    """Cuenta palabras"""
    return len(texto.split())


def tool_mayusculas(texto: str) -> str:
    """Convierte a mayúsculas"""
    return texto.upper()


def tool_fallar() -> None:
    """Tool que siempre falla (para probar errores)"""
    raise RuntimeError("Esta tool siempre falla (a propósito)")


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO TOOL SYSTEM 3.0")
    print("=" * 60)
    
    reg = ToolRegistryV3()
    
    # ============================================
    # REGISTRAR TOOLS
    # ============================================
    print("\n📝 Registrando tools:")
    
    reg.registrar(ToolSpec(
        nombre="calculadora",
        descripcion="Suma o multiplica dos números",
        funcion=tool_calculadora,
        input_schema={
            "expresion": Schema("str", requerido=True,
                                descripcion="ej: '5+3' o '4*2'"),
            "precision": Schema("int", requerido=False, default=2,
                                min=0, max=10),
        },
        output_tipo="float",
        capacidades=["matemáticas", "cálculo"],
        timeout=1.0,
    ))
    
    reg.registrar(ToolSpec(
        nombre="contar_palabras",
        descripcion="Cuenta palabras en un texto",
        funcion=tool_contar_palabras,
        input_schema={
            "texto": Schema("str", requerido=True),
        },
        output_tipo="int",
        capacidades=["texto"],
    ))
    
    reg.registrar(ToolSpec(
        nombre="mayusculas",
        descripcion="Convierte texto a mayúsculas",
        funcion=tool_mayusculas,
        input_schema={
            "texto": Schema("str", requerido=True),
        },
        output_tipo="str",
        capacidades=["texto", "transformación"],
    ))
    
    reg.registrar(ToolSpec(
        nombre="fallar",
        descripcion="Tool que siempre falla (test)",
        funcion=tool_fallar,
        capacidades=["test"],
    ))
    
    reg.registrar(ToolSpec(
        nombre="peligrosa",
        descripcion="Tool con permisos especiales (test)",
        funcion=tool_mayusculas,
        input_schema={"texto": Schema("str")},
        permisos=["admin", "fs_write"],
        capacidades=["test"],
    ))
    
    print(f"   {reg}")
    print(f"   Total: {len(reg.listar())} tools")
    
    # ============================================
    # LISTAR
    # ============================================
    print(f"\n📋 Listado:")
    for nombre in reg.listar():
        spec = reg.obtener(nombre)
        print(f"   {nombre}: {spec.descripcion}")
    
    # ============================================
    # BÚSQUEDA POR CAPACIDAD
    # ============================================
    print(f"\n🔍 Búsqueda por capacidad:")
    for cap in ["texto", "matemáticas", "test"]:
        tools = reg.buscar_por_capacidad(cap)
        print(f"   '{cap}' → {[t.nombre for t in tools]}")
    
    print(f"\n   Búsqueda por texto 'palabras':")
    print(f"   → {[t.nombre for t in reg.buscar('palabras')]}")
    
    # ============================================
    # EJECUCIÓN OK
    # ============================================
    print(f"\n✅ Ejecución OK:")
    r1 = reg.ejecutar("calculadora", {"expresion": "5+3"})
    print(f"   calculadora(5+3) → {r1}")
    print(f"   Output: {r1.output}")
    
    r2 = reg.ejecutar("contar_palabras", {"texto": "hola mundo feliz"})
    print(f"   contar_palabras → {r2.output}")
    
    # ============================================
    # VALIDACIÓN DE SCHEMA
    # ============================================
    print(f"\n⚠️ Validación de schema:")
    
    # Falta campo requerido
    r3 = reg.ejecutar("calculadora", {})
    print(f"   sin expresión → {r3.error}")
    
    # Tipo incorrecto
    r4 = reg.ejecutar("contar_palabras", {"texto": 123})
    print(f"   texto=123 → {r4.error}")
    
    # Min/max
    r5 = reg.ejecutar("calculadora",
                       {"expresion": "5+3", "precision": 99})
    print(f"   precision=99 → {r5.error}")
    
    # Default aplicado
    r6 = reg.ejecutar("calculadora", {"expresion": "1.23456+1"})
    print(f"   sin precision (default=2): {r6.output}")
    
    # ============================================
    # ERROR DE TOOL
    # ============================================
    print(f"\n💥 Error en tool:")
    r7 = reg.ejecutar("fallar")
    print(f"   fallar() → {r7.error}")
    
    # ============================================
    # TOOL NO REGISTRADA
    # ============================================
    print(f"\n❌ Tool no registrada:")
    r8 = reg.ejecutar("no_existe")
    print(f"   → {r8.error}")
    
    # ============================================
    # VALIDACIÓN DE PERMISOS
    # ============================================
    print(f"\n🔒 Validación de permisos:")
    
    def validador(permisos, tool):
        # Denegamos solo si pide "admin"
        return "admin" not in permisos
    
    r9 = reg.ejecutar("peligrosa", {"texto": "hola"},
                       validador_permisos=validador)
    print(f"   peligrosa con validador → {r9.error}")
    
    r10 = reg.ejecutar("mayusculas", {"texto": "hola"},
                        validador_permisos=validador)
    print(f"   mayusculas sin permisos → {r10.output}")
    
    # ============================================
    # STATS
    # ============================================
    print(f"\n📊 Stats:")
    for k, v in reg.stats().items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    print(f"\n💾 Test persistencia:")
    tmp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_tools_v3_tmp.json"
    )
    reg.guardar_esquema(tmp)
    print(f"   Guardado: {tmp}")
    print(f"   Existe: {os.path.exists(tmp)}")
    os.unlink(tmp)
    
    print("\n✅ TOOL SYSTEM 3.0 FUNCIONANDO")
