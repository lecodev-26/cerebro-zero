"""
CEREBRO ZERO 1.0 - Agente completo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from datetime import datetime

# Importar todos los módulos
from memory.retrieval import MemoryRetrieval
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from reasoning.planner import Razonamiento
from tools.tool_system import ToolSystem
from training.continuous_learning import ContinuousLearning
from evaluation.self_evaluation import SelfEvaluation
from evaluation.security import SecuritySystem

class CerebroZero:
    def __init__(self):
        print("🧠 INICIALIZANDO CEREBRO ZERO 1.0...")
        
        # Componentes
        self.memoria = MemoryRetrieval()
        self.razonamiento = Razonamiento()
        self.herramientas = ToolSystem()
        self.aprendizaje = ContinuousLearning(archivo="cerebro_zero_learning.json")
        self.autoevaluacion = SelfEvaluation()
        self.seguridad = SecuritySystem()
        
        # Estado
        self.nombre = "Cerebro Zero"
        self.version = "1.0.0"
        self.fecha_creacion = "2026-09-08"
        self.historial = []
        self.conocimientos = {
            'nombre': self.nombre,
            'version': self.version,
            'creador': 'Manuel',
            'fecha': self.fecha_creacion
        }
        
        print(f"✅ {self.nombre} v{self.version} inicializado")
    
    def procesar(self, entrada):
        """
        Procesa una entrada con TODOS los componentes
        """
        print(f"\n🧠 {self.nombre} procesando: {entrada}")
        print("="*50)
        
        # 1. SEGURIDAD - Verificar entrada
        seguro, mensaje = self.seguridad.verificar(entrada)
        if not seguro:
            print(f"🔒 BLOQUEADO: {mensaje}")
            return f"🔒 Entrada bloqueada por seguridad: {mensaje}"
        
        # 2. HERRAMIENTAS - Detectar si requiere herramienta
        resultado_herramienta = self._detectar_herramienta(entrada)
        if resultado_herramienta:
            # Autoevaluar
            evaluacion = self.autoevaluacion.evaluar(entrada, resultado_herramienta)
            # Aprender
            self.aprendizaje.aprender(entrada, resultado_herramienta, correcto=True)
            return resultado_herramienta
        
        # 3. MEMORIA - Buscar respuesta en memoria
        respuesta_memoria = self._buscar_en_memoria(entrada)
        if respuesta_memoria:
            evaluacion = self.autoevaluacion.evaluar(entrada, respuesta_memoria)
            self.aprendizaje.aprender(entrada, respuesta_memoria, correcto=True)
            return respuesta_memoria
        
        # 4. RAZONAMIENTO - Planificar y ejecutar
        resultados = self.razonamiento.razonar(entrada)
        if resultados and resultados[0] and 'He recibido' not in resultados[0]:
            evaluacion = self.autoevaluacion.evaluar(entrada, resultados[0])
            if evaluacion['confianza'] >= 0.7:
                self.aprendizaje.aprender(entrada, resultados[0], correcto=True)
                return resultados[0]
        
        # 5. RESPUESTA POR DEFECTO
        respuesta_default = "No tengo información sobre eso. ¿Puedes enseñarme?"
        self.aprendizaje.aprender(entrada, respuesta_default, correcto=False)
        return respuesta_default
    
    def _detectar_herramienta(self, entrada):
        """Detecta si la entrada requiere una herramienta"""
        entrada_lower = entrada.lower()
        
        # Matemáticas
        if any(op in entrada for op in ['+', '-', '*', '/']):
            numeros = re.findall(r'[-+]?\d*\.?\d+', entrada)
            if len(numeros) >= 2:
                a, b = float(numeros[0]), float(numeros[1])
                if '+' in entrada: return f"{a} + {b} = {a+b}"
                if '-' in entrada: return f"{a} - {b} = {a-b}"
                if '*' in entrada: return f"{a} × {b} = {a*b}"
                if '/' in entrada and b != 0: return f"{a} ÷ {b} = {a/b:.2f}"
        
        if 'hora' in entrada_lower:
            return self.herramientas.ejecutar('hora')
        if 'fecha' in entrada_lower:
            return self.herramientas.ejecutar('fecha')
        if 'cuenta' in entrada_lower or 'contar' in entrada_lower:
            return self.herramientas.ejecutar('contar', entrada)
        
        return None
    
    def _buscar_en_memoria(self, entrada):
        """Busca una respuesta en la memoria episódica"""
        episodios = self.memoria.get_all_episodic()
        for ep in episodios:
            if entrada.lower() in ep.get('input', '').lower():
                return ep.get('output', '')
        return None
    
    def enseñar(self, entrada, respuesta):
        """Enseña una respuesta correcta"""
        self.memoria.add_experience(entrada, respuesta)
        self.aprendizaje.aprender(entrada, respuesta, correcto=True)
        print(f"🎓 Enseñado: {entrada} → {respuesta}")
    
    def estadisticas(self):
        """Muestra todas las estadísticas"""
        print("\n" + "="*50)
        print(f"📊 ESTADÍSTICAS DE {self.nombre} v{self.version}")
        print("="*50)
        
        self.aprendizaje.estadisticas()
        self.autoevaluacion.estadisticas()
        self.seguridad.estadisticas_seguridad()
        
        print(f"\n📝 Historial: {len(self.historial)} entradas")
        print(f"🧠 Conocimientos: {len(self.conocimientos)}")
    
    def resumen(self):
        """Muestra un resumen del agente"""
        print("\n" + "="*50)
        print(f"🧠 {self.nombre} v{self.version}")
        print("="*50)
        print(f"   Creador: {self.conocimientos['creador']}")
        print(f"   Fecha: {self.fecha_creacion}")
        print(f"   Componentes: Memoria, Razonamiento, Herramientas,")
        print(f"               Aprendizaje, Autoevaluación, Seguridad")
        print("="*50)
    
    def __repr__(self):
        return f"CerebroZero(v{self.version})"

def prueba_cerebro_zero():
    print("🧠 PROBANDO CEREBRO ZERO 1.0")
    print("="*50)
    
    cerebro = CerebroZero()
    
    # Enseñar
    print("\n🎓 Enseñando respuestas...")
    cerebro.enseñar("hola", "Hola, soy Cerebro Zero 1.0")
    cerebro.enseñar("adiós", "Hasta luego, ha sido un placer")
    cerebro.enseñar("qué es la ia", "La inteligencia artificial es un campo de la computación")
    
    # Procesar
    print("\n📝 Procesando entradas:")
    entradas = [
        "hola",
        "¿Cuánto es 5 + 3?",
        "¿Qué hora es?",
        "¿Qué fecha es hoy?",
        "qué es la ia",
        "adiós",
        "algo que no sé",
        "rm -rf /",
    ]
    
    for entrada in entradas:
        respuesta = cerebro.procesar(entrada)
        print(f"\n   >>> {entrada}")
        print(f"   <<< {respuesta}")
    
    # Estadísticas
    cerebro.estadisticas()
    cerebro.resumen()
    
    print("\n✅ CEREBRO ZERO 1.0 FUNCIONANDO!")

if __name__ == "__main__":
    prueba_cerebro_zero()
