"""
Procedural Memory — Almacén de procedimientos (cómo hacer cosas)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Callable
from datetime import datetime


@dataclass
class Step:
    """Un paso de un procedimiento"""
    accion: str               # ej: "calcular", "buscar", "verificar"
    parametros: dict = field(default_factory=dict)
    descripcion: str = ""
    
    def to_dict(self) -> dict:
        return {
            'accion': self.accion,
            'parametros': self.parametros,
            'descripcion': self.descripcion,
        }


@dataclass
class Procedimiento:
    """Un procedimiento aprendido"""
    nombre: str
    pasos: List[Step]
    trigger: str                # patrón de entrada que lo activa
    exitos: int = 0
    fallos: int = 0
    creado: float = field(default_factory=time.time)
    ultimo_uso: float = field(default_factory=time.time)
    
    @property
    def tasa_exito(self) -> float:
        total = self.exitos + self.fallos
        if total == 0:
            return 0.5
        return self.exitos / total
    
    @property
    def uso_total(self) -> int:
        return self.exitos + self.fallos
    
    def registrar_exito(self):
        self.exitos += 1
        self.ultimo_uso = time.time()
    
    def registrar_fallo(self):
        self.fallos += 1
        self.ultimo_uso = time.time()
    
    def to_dict(self) -> dict:
        return {
            'nombre': self.nombre,
            'trigger': self.trigger,
            'pasos': [s.to_dict() for s in self.pasos],
            'exitos': self.exitos,
            'fallos': self.fallos,
            'tasa_exito': self.tasa_exito,
            'uso_total': self.uso_total,
        }
    
    def __repr__(self):
        return f"Procedimiento('{self.nombre}', {len(self.pasos)} pasos, exito={self.tasa_exito:.2f})"


class ProceduralMemory:
    """
    Memoria procedural: aprende y reutiliza procedimientos.
    
    Uso:
        pm = ProceduralMemory()
        pm.aprender("sumar", trigger="sumar|\\+", pasos=[
            Step("parsear", {}, "Extraer números"),
            Step("calcular", {"op": "+"}, "Sumar"),
            Step("verificar", {}, "Comprobar resultado"),
        ])
        
        proc = pm.buscar("¿Cuánto es 5 + 3?")
        if proc:
            pm.ejecutar(proc, contexto={'numeros': [5, 3]})
    """
    
    def __init__(self, max_procedimientos: int = 50):
        self.max_procedimientos = max_procedimientos
        self.procedimientos: dict = {}  # nombre -> Procedimiento
        self.historial = []  # últimos procedimientos ejecutados
    
    def aprender(self, nombre: str, trigger: str,
                 pasos: List[Step], sobrescribir: bool = True) -> Procedimiento:
        """
        Aprende un procedimiento nuevo.
        
        Args:
            nombre: identificador único
            trigger: patrón regex que lo activa
            pasos: lista de Step
            sobrescribir: si ya existe, reemplazarlo
        """
        if nombre in self.procedimientos and not sobrescribir:
            return self.procedimientos[nombre]
        
        # Si estamos llenos, expulsar el menos usado
        if len(self.procedimientos) >= self.max_procedimientos and nombre not in self.procedimientos:
            self._expulsar_menos_usado()
        
        proc = Procedimiento(
            nombre=nombre,
            pasos=pasos,
            trigger=trigger,
        )
        self.procedimientos[nombre] = proc
        return proc
    
    def aprender_de_exito(self, entrada: str, contexto: dict,
                          pasos: List[Step], nombre: str = None) -> Procedimiento:
        """
        Aprende automáticamente de una ejecución exitosa.
        
        Útil para que el sistema registre sus propios procedimientos.
        """
        if nombre is None:
            # Generar nombre a partir del hash de la entrada
            nombre = f"proc_{hashlib.md5(entrada.encode()).hexdigest()[:8]}"
        
        # Trigger = la entrada literal (por ahora)
        trigger = self._extraer_trigger(entrada)
        
        proc = self.aprender(nombre, trigger, pasos, sobrescribir=False)
        proc.registrar_exito()
        
        return proc
    
    def buscar(self, entrada: str) -> Optional[Procedimiento]:
        """
        Busca un procedimiento que pueda manejar la entrada.
        Devuelve el más exitoso si hay varios.
        """
        import re
        
        candidatos = []
        for proc in self.procedimientos.values():
            try:
                if re.search(proc.trigger, entrada, re.IGNORECASE):
                    candidatos.append(proc)
            except re.error:
                continue
        
        if not candidatos:
            return None
        
        # Ordenar por tasa de éxito + uso
        candidatos.sort(
            key=lambda p: (p.tasa_exito, p.uso_total),
            reverse=True,
        )
        
        return candidatos[0]
    
    def ejecutar(self, proc: Procedimiento, contexto: dict,
                 ejecutores: dict = None) -> dict:
        """
        Ejecuta un procedimiento paso a paso.
        
        Args:
            proc: procedimiento a ejecutar
            contexto: estado inicial
            ejecutores: dict {accion: función} para ejecutar cada paso
        
        Returns:
            dict con resultado, contexto final, éxito
        """
        if ejecutores is None:
            ejecutores = {}
        
        resultado = {
            'procedimiento': proc.nombre,
            'pasos_ejecutados': [],
            'contexto_final': contexto.copy(),
            'exito': False,
            'error': None,
        }
        
        ctx = contexto.copy()
        
        try:
            for i, paso in enumerate(proc.pasos):
                # Buscar ejecutor
                if paso.accion in ejecutores:
                    salida = ejecutores[paso.accion](ctx, paso.parametros)
                    # Actualizar contexto con la salida si es dict
                    if isinstance(salida, dict):
                        ctx.update(salida)
                    resultado['pasos_ejecutados'].append({
                        'paso': i,
                        'accion': paso.accion,
                        'salida': salida,
                    })
                else:
                    # No hay ejecutor: solo registramos
                    resultado['pasos_ejecutados'].append({
                        'paso': i,
                        'accion': paso.accion,
                        'salida': None,
                        'nota': 'sin ejecutor',
                    })
            
            resultado['exito'] = True
            resultado['contexto_final'] = ctx
            proc.registrar_exito()
        
        except Exception as e:
            resultado['exito'] = False
            resultado['error'] = str(e)
            proc.registrar_fallo()
        
        self.historial.append({
            'timestamp': time.time(),
            'procedimiento': proc.nombre,
            'exito': resultado['exito'],
        })
        
        if len(self.historial) > 100:
            self.historial = self.historial[-100:]
        
        return resultado
    
    def olvidar(self, nombre: str) -> bool:
        """Elimina un procedimiento"""
        if nombre in self.procedimientos:
            del self.procedimientos[nombre]
            return True
        return False
    
    def listar(self) -> List[dict]:
        """Lista todos los procedimientos ordenados por éxito"""
        procs = sorted(
            self.procedimientos.values(),
            key=lambda p: (p.tasa_exito, p.uso_total),
            reverse=True,
        )
        return [p.to_dict() for p in procs]
    
    def _extraer_trigger(self, entrada: str) -> str:
        """Extrae un patrón regex de la entrada"""
        import re
        # Escapar caracteres especiales
        trigger = re.escape(entrada.lower())
        # Pero permitir variaciones en números
        trigger = re.sub(r'\d+', r'\\d+', trigger)
        return trigger
    
    def _expulsar_menos_usado(self):
        """Expulsa el procedimiento menos usado"""
        if not self.procedimientos:
            return
        menos = min(
            self.procedimientos.values(),
            key=lambda p: (p.uso_total, p.tasa_exito),
        )
        del self.procedimientos[menos.nombre]
    
    def clear(self):
        self.procedimientos.clear()
    
    def size(self) -> int:
        return len(self.procedimientos)
    
    def resumen(self) -> dict:
        return {
            'size': self.size(),
            'max': self.max_procedimientos,
            'procedimientos': self.listar(),
            'historial_size': len(self.historial),
        }
    
    def __repr__(self):
        return f"ProceduralMemory({self.size()}/{self.max_procedimientos})"


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO PROCEDURAL MEMORY")
    print("="*60)
    
    pm = ProceduralMemory(max_procedimientos=5)
    
    # Aprender procedimiento para sumar
    print("\n📝 Aprendiendo 'sumar':")
    pm.aprender(
        nombre="sumar",
        trigger=r'suma|\+',
        pasos=[
            Step("parsear", {}, "Extraer números"),
            Step("calcular", {"op": "+"}, "Sumar números"),
            Step("verificar", {}, "Comprobar resultado"),
        ],
    )
    
    # Aprender procedimiento para multiplicar
    pm.aprender(
        nombre="multiplicar",
        trigger=r'multiplica|\*',
        pasos=[
            Step("parsear", {}, "Extraer números"),
            Step("calcular", {"op": "*"}, "Multiplicar números"),
            Step("verificar", {}, "Comprobar resultado"),
        ],
    )
    
    print(f"   {pm}")
    print(f"   Procedimientos: {[p.nombre for p in pm.procedimientos.values()]}")
    
    # Buscar procedimiento
    print(f"\n🔍 Buscar '¿Cuánto es 5 + 3?':")
    proc = pm.buscar("¿Cuánto es 5 + 3?")
    if proc:
        print(f"   ✅ Encontrado: {proc.nombre}")
    else:
        print(f"   ❌ No encontrado")
    
    print(f"\n🔍 Buscar '¿Cuánto es 10 * 7?':")
    proc = pm.buscar("¿Cuánto es 10 * 7?")
    if proc:
        print(f"   ✅ Encontrado: {proc.nombre}")
    else:
        print(f"   ❌ No encontrado")
    
    print(f"\n🔍 Buscar 'hola mundo':")
    proc = pm.buscar("hola mundo")
    if proc:
        print(f"   ✅ Encontrado: {proc.nombre}")
    else:
        print(f"   ❌ No encontrado (esperado)")
    
    # Ejecutar procedimiento
    print(f"\n⚙️ Ejecutar 'sumar' con contexto:")
    
    def parsear(ctx, params):
        import re
        nums = re.findall(r'-?\d+', ctx.get('entrada', ''))
        return {'numeros': [int(n) for n in nums]}
    
    def calcular(ctx, params):
        nums = ctx.get('numeros', [])
        op = params.get('op', '+')
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            if op == '+': r = a + b
            elif op == '*': r = a * b
            else: r = None
            return {'resultado': r}
        return {'resultado': None}
    
    def verificar(ctx, params):
        r = ctx.get('resultado')
        return {'verificado': r is not None}
    
    ejecutores = {
        'parsear': parsear,
        'calcular': calcular,
        'verificar': verificar,
    }
    
    proc = pm.procedimientos['sumar']
    resultado = pm.ejecutar(
        proc,
        contexto={'entrada': '¿Cuánto es 5 + 3?'},
        ejecutores=ejecutores,
    )
    
    print(f"   Éxito: {resultado['exito']}")
    print(f"   Resultado: {resultado['contexto_final'].get('resultado')}")
    print(f"   Pasos ejecutados: {len(resultado['pasos_ejecutados'])}")
    
    # Verificar tasa de éxito
    print(f"\n📊 Tasa de éxito de 'sumar': {proc.tasa_exito:.2f}")
    
    # Aprender de éxito
    print(f"\n🎓 Aprender de éxito:")
    nuevo = pm.aprender_de_exito(
        entrada="¿Cuánto es 100 - 50?",
        contexto={},
        pasos=[
            Step("parsear", {}, "Extraer números"),
            Step("calcular", {"op": "-"}, "Restar"),
        ],
    )
    print(f"   {nuevo}")
    
    # Resumen
    print(f"\n📊 Resumen:")
    resumen = pm.resumen()
    print(f"   Size: {resumen['size']}/{resumen['max']}")
    for p in resumen['procedimientos']:
        print(f"   - {p['nombre']}: éxito={p['tasa_exito']:.2f}, uso={p['uso_total']}")
    
    print("\n✅ PROCEDURAL MEMORY FUNCIONANDO")
