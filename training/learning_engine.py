"""
Learning Engine — Aprendizaje continuo con verificación
========================================================

Filosofía:
    Experiencia → Buffer → Replay → Evaluación → Accept/Reject

Regla de oro:
    El modelo NUNCA se actualiza automáticamente.
    Solo si mejora objetivamente respecto al anterior.

Componentes:
- Experiencia: state, action, result, reward, error, verified
- ExperienceBuffer: almacén con límite y muestreo
- Evaluador de mejora: compara viejo vs nuevo
- LearningEngine: orquesta todo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime


# ============================================
# EXPERIENCIA
# ============================================

@dataclass
class Experiencia:
    """Una experiencia vivida por el agente"""
    estado: Any
    accion: Any
    resultado: Any
    recompensa: float = 0.0
    error: Optional[str] = None
    verificado: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "estado": str(self.estado)[:200],
            "accion": str(self.accion)[:200],
            "resultado": str(self.resultado)[:200],
            "recompensa": self.recompensa,
            "error": self.error,
            "verificado": self.verificado,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ============================================
# BUFFER DE EXPERIENCIAS
# ============================================

class ExperienceBuffer:
    """
    Almacén circular de experiencias con muestreo.
    
    Uso:
        buf = ExperienceBuffer(capacidad=1000)
        buf.add(exp)
        lote = buf.sample(32)
    """
    
    def __init__(self, capacidad: int = 1000):
        self.capacidad = capacidad
        self.experiencias: List[Experiencia] = []
    
    def add(self, exp: Experiencia):
        self.experiencias.append(exp)
        if len(self.experiencias) > self.capacidad:
            # Eliminar las más antiguas
            self.experiencias = self.experiencias[-self.capacidad:]
    
    def sample(self, n: int) -> List[Experiencia]:
        if not self.experiencias:
            return []
        n = min(n, len(self.experiencias))
        return random.sample(self.experiencias, n)
    
    def sample_positivas(self, n: int) -> List[Experiencia]:
        """Muestrear solo experiencias con recompensa > 0"""
        positivas = [e for e in self.experiencias if e.recompensa > 0]
        if not positivas:
            return []
        n = min(n, len(positivas))
        return random.sample(positivas, n)
    
    def clear(self):
        self.experiencias = []
    
    def __len__(self):
        return len(self.experiencias)
    
    def stats(self) -> dict:
        if not self.experiencias:
            return {"total": 0}
        
        recompensas = [e.recompensa for e in self.experiencias]
        return {
            "total": len(self.experiencias),
            "capacidad": self.capacidad,
            "recompensa_media": sum(recompensas) / len(recompensas),
            "recompensa_max": max(recompensas),
            "recompensa_min": min(recompensas),
            "verificadas": sum(1 for e in self.experiencias if e.verificado),
            "con_error": sum(1 for e in self.experiencias if e.error),
        }
    
    def __repr__(self):
        return f"ExperienceBuffer({len(self)}/{self.capacidad})"


# ============================================
# EVALUADOR DE MEJORA
# ============================================

@dataclass
class ResultadoEvaluacion:
    """Resultado de comparar dos versiones"""
    aceptar: bool
    mejora: float
    razon: str
    score_viejo: float
    score_nuevo: float
    
    def to_dict(self) -> dict:
        return {
            "aceptar": self.aceptar,
            "mejora": self.mejora,
            "razon": self.razon,
            "score_viejo": self.score_viejo,
            "score_nuevo": self.score_nuevo,
        }


class EvaluadorMejora:
    """
    Decide si una actualización se acepta o rechaza.
    
    Regla:
        Si score_nuevo > score_viejo + umbral → ACEPTAR
        Si no → RECHAZAR
    """
    
    def __init__(self, umbral: float = 0.0):
        self.umbral = umbral
    
    def evaluar(self, score_viejo: float, score_nuevo: float) -> ResultadoEvaluacion:
        mejora = score_nuevo - score_viejo
        aceptar = mejora > self.umbral
        
        if aceptar:
            razon = f"Mejora de {mejora:+.4f} supera umbral {self.umbral}"
        else:
            razon = f"Mejora {mejora:+.4f} no supera umbral {self.umbral}"
        
        return ResultadoEvaluacion(
            aceptar=aceptar,
            mejora=mejora,
            razon=razon,
            score_viejo=score_viejo,
            score_nuevo=score_nuevo,
        )


# ============================================
# LEARNING ENGINE
# ============================================

class LearningEngine:
    """
    Motor de aprendizaje con:
    - Registro de experiencias
    - Replay para entrenamiento
    - Evaluación accept/reject
    - Historial de decisiones
    """
    
    def __init__(
        self,
        buffer_capacidad: int = 1000,
        umbral_mejora: float = 0.0,
    ):
        self.buffer = ExperienceBuffer(capacidad=buffer_capacidad)
        self.evaluador = EvaluadorMejora(umbral=umbral_mejora)
        self.historial_evaluaciones: List[ResultadoEvaluacion] = []
        self.versiones: List[dict] = []  # versiones aceptadas
        self.version_actual: dict = {"version": 0, "score": 0.0}
    
    # ============================================
    # REGISTRO DE EXPERIENCIAS
    # ============================================
    
    def registrar_experiencia(
        self,
        estado: Any,
        accion: Any,
        resultado: Any,
        recompensa: float = 0.0,
        error: Optional[str] = None,
        verificado: bool = False,
        metadata: Optional[dict] = None,
    ) -> Experiencia:
        exp = Experiencia(
            estado=estado,
            accion=accion,
            resultado=resultado,
            recompensa=recompensa,
            error=error,
            verificado=verificado,
            metadata=metadata or {},
        )
        self.buffer.add(exp)
        return exp
    
    def registrar_exito(self, estado, accion, resultado, recompensa=1.0, **kw):
        return self.registrar_experiencia(
            estado, accion, resultado,
            recompensa=recompensa, verificado=True, **kw
        )
    
    def registrar_fallo(self, estado, accion, error, recompensa=-0.5, **kw):
        return self.registrar_experiencia(
            estado, accion, None,
            recompensa=recompensa, error=error, **kw
        )
    
    # ============================================
    # REPLAY
    # ============================================
    
    def sample_replay(self, n: int = 32) -> List[Experiencia]:
        """Muestrear n experiencias para entrenar"""
        return self.buffer.sample(n)
    
    def sample_exitos(self, n: int = 32) -> List[Experiencia]:
        """Muestrear experiencias exitosas"""
        return self.buffer.sample_positivas(n)
    
    # ============================================
    # EVALUACIÓN ACCEPT/REJECT
    # ============================================
    
    def evaluar_mejora(self, score_nuevo: float) -> ResultadoEvaluacion:
        """
        Evalúa si la nueva versión mejora respecto a la actual.
        NO actualiza nada — solo evalúa.
        """
        resultado = self.evaluador.evaluar(
            self.version_actual["score"],
            score_nuevo,
        )
        self.historial_evaluaciones.append(resultado)
        return resultado
    
    def intentar_actualizar(self, score_nuevo: float) -> ResultadoEvaluacion:
        """
        Intenta actualizar la versión actual.
        Si mejora → ACEPTA y actualiza.
        Si no → RECHAZA y no cambia nada.
        """
        resultado = self.evaluar_mejora(score_nuevo)
        
        if resultado.aceptar:
            self.version_actual = {
                "version": self.version_actual["version"] + 1,
                "score": score_nuevo,
                "timestamp": time.time(),
            }
            self.versiones.append(self.version_actual.copy())
        
        return resultado
    
    # ============================================
    # ENTRENAMIENTO
    # ============================================
    
    def entrenar_con_replay(
        self,
        modelo_entrenar: Callable,
        evaluar_modelo: Callable,
        n_replay: int = 32,
    ) -> dict:
        """
        Entrena con replay y decide si aceptar la actualización.
        
        Args:
            modelo_entrenar: función(lote) que entrena el modelo
            evaluar_modelo: función() que devuelve un score del modelo
            n_replay: número de experiencias a usar
        
        Returns:
            dict con resultado del intento
        """
        # Score antes
        score_antes = evaluar_modelo()
        
        # Replay
        lote = self.sample_replay(n_replay)
        if not lote:
            return {
                "ok": False,
                "razon": "Buffer vacío",
                "score_antes": score_antes,
            }
        
        # Entrenar
        try:
            modelo_entrenar(lote)
        except Exception as e:
            return {
                "ok": False,
                "razon": f"Error entrenando: {e}",
                "score_antes": score_antes,
            }
        
        # Score después
        score_despues = evaluar_modelo()
        
        # Decidir
        resultado = self.intentar_actualizar(score_despues)
        
        return {
            "ok": resultado.aceptar,
            "razon": resultado.razon,
            "score_antes": score_antes,
            "score_despues": score_despues,
            "mejora": resultado.mejora,
            "version": self.version_actual["version"],
        }
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def guardar(self, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        data = {
            "version_actual": self.version_actual,
            "versiones": self.versiones,
            "buffer_stats": self.buffer.stats(),
            "experiencias": [e.to_dict() for e in self.buffer.experiencias[-100:]],
            "historial_evaluaciones": [
                r.to_dict() for r in self.historial_evaluaciones[-50:]
            ],
            "guardado": datetime.now().isoformat(),
        }
        with open(archivo, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    def resumen(self) -> dict:
        return {
            "buffer": self.buffer.stats(),
            "version_actual": self.version_actual,
            "versiones_aceptadas": len(self.versiones),
            "evaluaciones_totales": len(self.historial_evaluaciones),
            "ultima_evaluacion": (
                self.historial_evaluaciones[-1].to_dict()
                if self.historial_evaluaciones else None
            ),
        }
    
    def __repr__(self):
        return (
            f"LearningEngine(v{self.version_actual['version']}, "
            f"buffer={len(self.buffer)}, "
            f"score={self.version_actual['score']:.3f})"
        )


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO LEARNING ENGINE")
    print("=" * 60)
    
    engine = LearningEngine(buffer_capacidad=100, umbral_mejora=0.0)
    
    # Registrar experiencias
    print("\n📝 Registrando experiencias:")
    engine.registrar_exito("estado1", "sumar", "resultado=8", recompensa=1.0)
    engine.registrar_exito("estado2", "multiplicar", "resultado=42", recompensa=1.0)
    engine.registrar_fallo("estado3", "dividir", "división por cero")
    engine.registrar_experiencia("estado4", "esperar", "nada", recompensa=0.0)
    print(f"   Buffer: {engine.buffer}")
    print(f"   Stats: {engine.buffer.stats()}")
    
    # Sample replay
    print(f"\n🎲 Sample replay(3):")
    lote = engine.sample_replay(3)
    print(f"   {len(lote)} experiencias muestreadas")
    
    print(f"\n✅ Sample exitos(5):")
    exitos = engine.sample_exitos(5)
    print(f"   {len(exitos)} exitos muestreados")
    
    # Evaluación accept/reject
    print(f"\n⚖️ Evaluación accept/reject:")
    
    # Primera: v0 score=0, nuevo=0.5 → ACCEPT
    r1 = engine.intentar_actualizar(0.5)
    print(f"   0.0 → 0.5: {'✅ ACEPTAR' if r1.aceptar else '❌ RECHAZAR'} ({r1.razon})")
    
    # Segunda: v1 score=0.5, nuevo=0.8 → ACCEPT
    r2 = engine.intentar_actualizar(0.8)
    print(f"   0.5 → 0.8: {'✅ ACEPTAR' if r2.aceptar else '❌ RECHAZAR'}")
    
    # Tercera: v2 score=0.8, nuevo=0.6 → RECHAZAR
    r3 = engine.intentar_actualizar(0.6)
    print(f"   0.8 → 0.6: {'✅ ACEPTAR' if r3.aceptar else '❌ RECHAZAR'}")
    
    # Cuarta: v2 score=0.8, nuevo=0.8 → RECHAZAR (no mejora)
    r4 = engine.intentar_actualizar(0.8)
    print(f"   0.8 → 0.8: {'✅ ACEPTAR' if r4.aceptar else '❌ RECHAZAR'}")
    
    # Ver versión actual
    print(f"\n📌 Versión final:")
    print(f"   {engine.version_actual}")
    print(f"   Versiones aceptadas: {len(engine.versiones)}")
    
    # Entrenamiento con replay
    print(f"\n🏋️ Test entrenar_con_replay:")
    
    contador = {"n": 0}
    def modelo_entrenar(lote):
        contador["n"] += 1
    
    def evaluar_modelo():
        # Score simulado que va subiendo
        return 0.9 + contador["n"] * 0.01
    
    resultado = engine.entrenar_con_replay(modelo_entrenar, evaluar_modelo, n_replay=5)
    print(f"   Resultado: {resultado}")
    
    # Resumen
    print(f"\n📊 Resumen:")
    r = engine.resumen()
    print(f"   Version actual: {r['version_actual']}")
    print(f"   Versiones aceptadas: {r['versiones_aceptadas']}")
    print(f"   Evaluaciones totales: {r['evaluaciones_totales']}")
    
    # Persistencia
    print(f"\n💾 Test persistencia:")
    tmp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_learning_tmp.json"
    )
    engine.guardar(tmp)
    print(f"   Guardado en: {tmp}")
    print(f"   Existe: {os.path.exists(tmp)}")
    os.unlink(tmp)
    
    print("\n✅ LEARNING ENGINE FUNCIONANDO")
    print(f"   {engine}")
