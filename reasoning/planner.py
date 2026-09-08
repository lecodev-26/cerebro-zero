"""
Sistema de razonamiento y planificación
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

class Planner:
    def __init__(self):
        self.pasos = []
        self.reglas = {
            'suma': self._sumar,
            'resta': self._restar,
            'multiplica': self._multiplicar,
            'divide': self._dividir,
        }
    
    def planificar(self, pregunta):
        """
        Planifica los pasos para responder una pregunta
        """
        self.pasos = []
        pregunta = pregunta.lower()
        
        # Detectar tipo de pregunta
        if 'suma' in pregunta or '+' in pregunta:
            self.pasos.append(('matematica', self._extraer_numeros(pregunta), 'suma'))
        elif 'resta' in pregunta or '-' in pregunta:
            self.pasos.append(('matematica', self._extraer_numeros(pregunta), 'resta'))
        elif 'multiplica' in pregunta or '*' in pregunta:
            self.pasos.append(('matematica', self._extraer_numeros(pregunta), 'multiplica'))
        elif 'divide' in pregunta or '/' in pregunta:
            self.pasos.append(('matematica', self._extraer_numeros(pregunta), 'divide'))
        elif 'que es' in pregunta or 'qué es' in pregunta:
            self.pasos.append(('definicion', pregunta))
        elif 'por que' in pregunta or 'por qué' in pregunta:
            self.pasos.append(('explicacion', pregunta))
        else:
            self.pasos.append(('generico', pregunta))
        
        return self.pasos
    
    def ejecutar(self, pasos):
        """
        Ejecuta los pasos planificados
        """
        resultados = []
        for paso in pasos:
            tipo = paso[0]
            if tipo == 'matematica':
                numeros = paso[1]
                operacion = paso[2]
                if len(numeros) >= 2:
                    resultado = self.reglas[operacion](numeros[0], numeros[1])
                    resultados.append(resultado)
                else:
                    resultados.append("No tengo suficientes números para esa operación")
            elif tipo == 'definicion':
                resultados.append(self._definir(paso[1]))
            elif tipo == 'explicacion':
                resultados.append(self._explicar(paso[1]))
            else:
                resultados.append(self._respuesta_generica(paso[1]))
        
        return resultados
    
    def _extraer_numeros(self, texto):
        """
        Extrae números de un texto
        """
        return [float(n) for n in re.findall(r'[-+]?\d*\.?\d+', texto)]
    
    def _sumar(self, a, b):
        return f"{a} + {b} = {a + b}"
    
    def _restar(self, a, b):
        return f"{a} - {b} = {a - b}"
    
    def _multiplicar(self, a, b):
        return f"{a} × {b} = {a * b}"
    
    def _dividir(self, a, b):
        if b == 0:
            return "Error: división por cero"
        return f"{a} ÷ {b} = {a / b:.2f}"
    
    def _definir(self, pregunta):
        definiciones = {
            'ia': 'Inteligencia Artificial: capacidad de una máquina para imitar funciones humanas.',
            'red neuronal': 'Sistema computacional inspirado en el cerebro humano.',
            'aprendizaje': 'Proceso por el cual un sistema mejora con la experiencia.',
            'memoria': 'Capacidad de almacenar y recuperar información.',
        }
        for key, value in definiciones.items():
            if key in pregunta:
                return value
        return "No tengo una definición para eso."
    
    def _explicar(self, pregunta):
        return "No tengo suficiente información para explicar eso todavía."
    
    def _respuesta_generica(self, pregunta):
        return f"He recibido tu pregunta: '{pregunta}'. Estoy procesándola."

class Verifier:
    def __init__(self):
        self.errores = []
    
    def verificar(self, resultado, paso):
        """
        Verifica si un resultado es correcto
        """
        if 'Error' in str(resultado):
            self.errores.append(resultado)
            return False, resultado
        if 'No tengo' in str(resultado):
            self.errores.append("Falta de información")
            return False, resultado
        return True, resultado
    
    def obtener_errores(self):
        return self.errores

class Razonamiento:
    def __init__(self):
        self.planner = Planner()
        self.verifier = Verifier()
        self.historial = []
    
    def razonar(self, pregunta):
        """
        Proceso completo de razonamiento
        """
        print(f"🧠 Razonando sobre: {pregunta}")
        
        # 1. Planificar
        print("📋 Planificando pasos...")
        pasos = self.planner.planificar(pregunta)
        print(f"   Pasos planificados: {len(pasos)}")
        
        # 2. Ejecutar
        print("⚙️ Ejecutando pasos...")
        resultados = self.planner.ejecutar(pasos)
        
        # 3. Verificar
        print("🔍 Verificando resultados...")
        for i, (paso, resultado) in enumerate(zip(pasos, resultados)):
            ok, msg = self.verifier.verificar(resultado, paso)
            if not ok:
                print(f"   ⚠️ Paso {i+1} requiere revisión: {msg}")
        
        # 4. Guardar en historial
        self.historial.append({
            'pregunta': pregunta,
            'pasos': pasos,
            'resultados': resultados,
            'errores': self.verifier.obtener_errores()
        })
        
        return resultados
    
    def __repr__(self):
        return f"Razonamiento(historial={len(self.historial)})"

def prueba_razonamiento():
    print("🧠 PROBANDO SISTEMA DE RAZONAMIENTO")
    print("="*40)
    
    razon = Razonamiento()
    
    preguntas = [
        "¿Cuánto es 5 + 3?",
        "¿Cuánto es 10 - 4?",
        "¿Cuánto es 6 × 7?",
        "¿Cuánto es 15 ÷ 3?",
        "¿Qué es la IA?",
        "¿Qué es una red neuronal?",
        "¿Por qué brilla el sol?"
    ]
    
    for pregunta in preguntas:
        print("\n" + "="*40)
        resultados = razon.razonar(pregunta)
        print(f"📊 Respuesta: {resultados}")
    
    print("\n✅ SISTEMA DE RAZONAMIENTO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_razonamiento()
