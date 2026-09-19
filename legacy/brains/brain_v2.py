"""
Agente con aprendizaje continuo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from training.continuous_learning import ContinuousLearning
from memory.retrieval import MemoryRetrieval
from reasoning.planner import Razonamiento
from tools.tool_system import ToolSystem

class AgenteV2:
    def __init__(self, nombre="Cerebro Zero"):
        self.nombre = nombre
        self.memoria = MemoryRetrieval()
        self.razonamiento = Razonamiento()
        self.herramientas = ToolSystem()
        self.aprendizaje = ContinuousLearning()
        self.historial = []
        self.conocimientos = {
            'nombre': nombre,
            'creador': 'Manuel',
            'version': '2.0.0'
        }
    
    def procesar(self, entrada):
        """Procesa una entrada con aprendizaje continuo"""
        print(f"\n🧠 {self.nombre} procesando: {entrada}")
        
        # 1. Intentar herramientas
        resultado = self._detectar_herramienta(entrada)
        if resultado:
            self.aprendizaje.aprender(entrada, resultado, correcto=True)
            return resultado
        
        # 2. Intentar memoria
        respuesta_memoria = self._buscar_memoria(entrada)
        if respuesta_memoria:
            self.aprendizaje.aprender(entrada, respuesta_memoria, correcto=True)
            return respuesta_memoria
        
        # 3. Intentar razonamiento
        resultados = self.razonamiento.razonar(entrada)
        if resultados and resultados[0] and 'He recibido' not in resultados[0]:
            self.aprendizaje.aprender(entrada, resultados[0], correcto=True)
            return resultados[0]
        
        # 4. Respuesta por defecto
        respuesta_default = "No tengo información sobre eso. ¿Puedes enseñarme?"
        self.aprendizaje.aprender(entrada, respuesta_default, correcto=False)
        return respuesta_default
    
    def _detectar_herramienta(self, entrada):
        """Detecta herramientas"""
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
        
        return None
    
    def _buscar_memoria(self, entrada):
        """Busca en memoria episódica"""
        episodios = self.memoria.get_all_episodic()
        for ep in episodios:
            if entrada.lower() in ep.get('input', '').lower():
                return ep.get('output', '')
        return None
    
    def enseñar(self, entrada, respuesta_correcta):
        """Enseña una respuesta correcta"""
        # Guardar en memoria
        self.memoria.add_experience(entrada, respuesta_correcta)
        # Guardar en aprendizaje
        self.aprendizaje.aprender(entrada, respuesta_correcta, correcto=True)
        print(f"🎓 Enseñado: {entrada} → {respuesta_correcta}")
    
    def estadisticas(self):
        """Muestra estadísticas"""
        self.aprendizaje.estadisticas()
    
    def __repr__(self):
        return f"AgenteV2({self.nombre})"

def prueba_agente_v2():
    print("🧠 PROBANDO AGENTE CON APRENDIZAJE CONTINUO")
    print("="*50)
    
    agente = AgenteV2()
    
    # Enseñar respuestas
    print("\n🎓 Enseñando al agente...")
    agente.enseñar("hola", "Hola, soy Cerebro Zero")
    agente.enseñar("adiós", "Hasta luego, ha sido un placer")
    agente.enseñar("qué es la ia", "La inteligencia artificial es un campo de la computación")
    
    # Probar procesamiento
    print("\n📝 Procesando entradas:")
    entradas = [
        "hola",
        "¿Cuánto es 5 + 3?",
        "¿Qué hora es?",
        "qué es la ia",
        "adiós",
        "algo que no sé"
    ]
    
    for entrada in entradas:
        respuesta = agente.procesar(entrada)
        print(f"\n   >>> {entrada}")
        print(f"   <<< {respuesta}")
    
    # Estadísticas
    agente.estadisticas()
    
    print("\n✅ AGENTE CON APRENDIZAJE CONTINUO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_agente_v2()
