"""
Agente completo - Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from memory.retrieval import MemoryRetrieval
from reasoning.planner import Razonamiento
from tools.tool_system import ToolSystem
from experiments.memory_attention import MemoryAttention

class Agente:
    def __init__(self, nombre="Cerebro Zero"):
        self.nombre = nombre
        self.memoria = MemoryRetrieval()
        self.razonamiento = Razonamiento()
        self.herramientas = ToolSystem()
        self.memory_attention = MemoryAttention()
        self.historial = []
        self.conocimientos = {
            'nombre': nombre,
            'creador': 'Manuel',
            'version': '1.0.0',
            'fecha_creacion': '2026-09-08'
        }
    
    def procesar(self, entrada):
        """
        Procesa una entrada completa
        """
        print(f"\n🧠 {self.nombre} procesando: {entrada}")
        print("="*40)
        
        # 1. Guardar en historial
        self.historial.append(entrada)
        
        # 2. Intentar usar herramientas directamente
        resultado_herramienta = self._detectar_herramienta(entrada)
        if resultado_herramienta:
            return resultado_herramienta
        
        # 3. Usar memoria + atención
        resultado_memoria = self.memory_attention.process(entrada)
        if resultado_memoria['fuente'] != 'default':
            return resultado_memoria['respuesta']
        
        # 4. Usar razonamiento
        resultados_razon = self.razonamiento.razonar(entrada)
        if resultados_razon and resultados_razon[0]:
            return resultados_razon[0]
        
        # 5. Respuesta por defecto
        return "No tengo suficiente información para responder a eso."
    
    def _detectar_herramienta(self, entrada):
        """
        Detecta si la entrada requiere una herramienta
        """
        entrada_lower = entrada.lower()
        
        # Operaciones matemáticas
        if any(op in entrada_lower for op in ['suma', 'resta', 'multiplica', 'divide', '+', '-', '*', '/']):
            numeros = re.findall(r'[-+]?\d*\.?\d+', entrada)
            if len(numeros) >= 2:
                if '+' in entrada or 'suma' in entrada_lower:
                    return f"{numeros[0]} + {numeros[1]} = {float(numeros[0]) + float(numeros[1])}"
                elif '-' in entrada or 'resta' in entrada_lower:
                    return f"{numeros[0]} - {numeros[1]} = {float(numeros[0]) - float(numeros[1])}"
                elif '*' in entrada or 'multiplica' in entrada_lower:
                    return f"{numeros[0]} × {numeros[1]} = {float(numeros[0]) * float(numeros[1])}"
                elif '/' in entrada or 'divide' in entrada_lower:
                    if float(numeros[1]) != 0:
                        return f"{numeros[0]} ÷ {numeros[1]} = {float(numeros[0]) / float(numeros[1]):.2f}"
                    return "Error: división por cero"
        
        # Fecha y hora
        if 'fecha' in entrada_lower:
            return self.herramientas.ejecutar('fecha')
        if 'hora' in entrada_lower:
            return self.herramientas.ejecutar('hora')
        
        return None
    
    def aprender(self, entrada, salida):
        """
        Aprende de una experiencia
        """
        self.memory_attention.learn(entrada, salida)
        self.memoria.add_experience(entrada, salida)
        print(f"🧠 Aprendido: {entrada} → {salida}")
    
    def recordar(self, consulta):
        """
        Recuerda información relacionada
        """
        return self.memoria.remember(consulta)
    
    def resumen(self):
        """
        Resumen del estado del agente
        """
        print(f"\n🧠 RESUMEN DE {self.nombre}")
        print("="*40)
        print(f"📊 Memoria: {self.memoria}")
        print(f"🔧 Herramientas: {self.herramientas.listar()}")
        print(f"📝 Historial: {len(self.historial)} entradas")
        print(f"🧠 Conocimientos: {len(self.conocimientos)}")

def prueba_agente():
    print("🧠 PROBANDO AGENTE COMPLETO")
    print("="*40)
    
    agente = Agente()
    
    print("\n📝 Pruebas de entrada:")
    entradas = [
        "hola",
        "¿Cuánto es 5 + 3?",
        "¿Qué hora es?",
        "¿Qué fecha es hoy?",
        "¿Qué es la IA?",
        "adiós",
        "¿Cuánto es 10 * 7?",
    ]
    
    for entrada in entradas:
        respuesta = agente.procesar(entrada)
        print(f"\n   >>> Entrada: {entrada}")
        print(f"   <<< Respuesta: {respuesta}")
    
    print("\n📊 Resumen del agente:")
    agente.resumen()
    
    print("\n✅ AGENTE COMPLETO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_agente()
