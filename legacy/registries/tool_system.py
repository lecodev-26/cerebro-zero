"""
Sistema de herramientas para Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import datetime
import math

class ToolSystem:
    def __init__(self):
        self.tools = {
            'calculadora': self.calculadora,
            'fecha': self.fecha,
            'hora': self.hora,
            'contar': self.contar,
            'mayusculas': self.mayusculas,
            'minusculas': self.minusculas,
            'longitud': self.longitud,
            'invertir': self.invertir,
            'sumar_lista': self.sumar_lista,
            'promedio': self.promedio,
        }
        self.historial = []
    
    def ejecutar(self, comando, *args):
        """
        Ejecuta una herramienta
        """
        if comando in self.tools:
            try:
                resultado = self.tools[comando](*args)
                self.historial.append({
                    'comando': comando,
                    'args': args,
                    'resultado': resultado
                })
                return resultado
            except Exception as e:
                return f"❌ Error: {e}"
        return f"❌ Herramienta desconocida: {comando}"
    
    # === HERRAMIENTAS ===
    
    def calculadora(self, expresion):
        """
        Calcula una expresión matemática simple
        """
        try:
            # Limpiar expresión
            expresion = expresion.replace('×', '*').replace('÷', '/')
            # Solo permitir números y operadores básicos
            if not re.match(r'^[\d+\-*/().\s]+$', expresion):
                return "❌ Expresión inválida"
            resultado = eval(expresion)
            return f"{expresion} = {resultado}"
        except:
            return "❌ Error al calcular"
    
    def fecha(self):
        return f"📅 {datetime.datetime.now().strftime('%d/%m/%Y')}"
    
    def hora(self):
        return f"🕐 {datetime.datetime.now().strftime('%H:%M:%S')}"
    
    def contar(self, texto):
        """
        Cuenta palabras en un texto
        """
        palabras = texto.split()
        return f"📊 {len(palabras)} palabras"
    
    def mayusculas(self, texto):
        return texto.upper()
    
    def minusculas(self, texto):
        return texto.lower()
    
    def longitud(self, texto):
        return f"📏 {len(texto)} caracteres"
    
    def invertir(self, texto):
        return texto[::-1]
    
    def sumar_lista(self, *numeros):
        """
        Suma una lista de números
        """
        try:
            numeros = [float(n) for n in numeros]
            return f"Suma: {sum(numeros)}"
        except:
            return "❌ Error: asegúrate de que todos son números"
    
    def promedio(self, *numeros):
        """
        Calcula el promedio de una lista de números
        """
        try:
            numeros = [float(n) for n in numeros]
            promedio = sum(numeros) / len(numeros)
            return f"Promedio: {promedio:.2f}"
        except:
            return "❌ Error: asegúrate de que todos son números"
    
    def listar(self):
        return list(self.tools.keys())
    
    def __repr__(self):
        return f"ToolSystem({len(self.tools)} herramientas)"

def prueba_tools():
    print("🧠 PROBANDO SISTEMA DE HERRAMIENTAS")
    print("="*40)
    
    tools = ToolSystem()
    
    print(f"🔧 Herramientas disponibles: {tools.listar()}")
    
    print("\n📝 Pruebas:")
    print(f"   calculadora('5 + 3'): {tools.ejecutar('calculadora', '5 + 3')}")
    print(f"   calculadora('10 * 7'): {tools.ejecutar('calculadora', '10 * 7')}")
    print(f"   fecha(): {tools.ejecutar('fecha')}")
    print(f"   hora(): {tools.ejecutar('hora')}")
    print(f"   contar('Hola mundo'): {tools.ejecutar('contar', 'Hola mundo')}")
    print(f"   mayusculas('hola'): {tools.ejecutar('mayusculas', 'hola')}")
    print(f"   invertir('cerebro'): {tools.ejecutar('invertir', 'cerebro')}")
    print(f"   sumar_lista(1,2,3,4): {tools.ejecutar('sumar_lista', 1, 2, 3, 4)}")
    print(f"   promedio(1,2,3,4,5): {tools.ejecutar('promedio', 1, 2, 3, 4, 5)}")
    
    print("\n✅ SISTEMA DE HERRAMIENTAS FUNCIONANDO!")

if __name__ == "__main__":
    prueba_tools()
