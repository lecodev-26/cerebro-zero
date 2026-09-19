import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import re

class Herramientas:
    def __init__(self):
        self.tools = {
            'sumar': self.sumar,
            'restar': self.restar,
            'multiplicar': self.multiplicar,
            'dividir': self.dividir,
            'potencia': self.potencia,
            'raiz': self.raiz,
            'buscar': self.buscar,
            'contar': self.contar
        }
        self.memoria = {}
    
    def sumar(self, a, b):
        return a + b
    
    def restar(self, a, b):
        return a - b
    
    def multiplicar(self, a, b):
        return a * b
    
    def dividir(self, a, b):
        if b == 0:
            return "Error: División por cero"
        return a / b
    
    def potencia(self, a, b):
        return a ** b
    
    def raiz(self, a, n=2):
        return a ** (1/n)
    
    def buscar(self, texto, patron):
        try:
            return re.findall(patron, texto)
        except:
            return "Error en la búsqueda"
    
    def contar(self, lista):
        return len(lista)
    
    def ejecutar(self, comando, *args):
        if comando in self.tools:
            try:
                return self.tools[comando](*args)
            except Exception as e:
                return f"Error al ejecutar {comando}: {e}"
        return f"Herramienta desconocida: {comando}"
    
    def listar(self):
        return list(self.tools.keys())
    
    def describir(self, comando):
        descripciones = {
            'sumar': 'Suma dos números: sumar(a, b)',
            'restar': 'Resta dos números: restar(a, b)',
            'multiplicar': 'Multiplica dos números: multiplicar(a, b)',
            'dividir': 'Divide dos números: dividir(a, b)',
            'potencia': 'Eleva a a la b: potencia(a, b)',
            'raiz': 'Calcula la raíz n-ésima de a: raiz(a, n=2)',
            'buscar': 'Busca un patrón en un texto: buscar(texto, patron)',
            'contar': 'Cuenta elementos en una lista: contar(lista)'
        }
        return descripciones.get(comando, "Herramienta no descrita")

def prueba_herramientas():
    print("🧠 PROBANDO HERRAMIENTAS")
    print("="*30)
    
    tools = Herramientas()
    print(f"Herramientas disponibles: {tools.listar()}")
    
    print(f"\n🔧 Ejecutando herramientas:")
    print(f"   sumar(5, 3) = {tools.ejecutar('sumar', 5, 3)}")
    print(f"   multiplicar(4, 7) = {tools.ejecutar('multiplicar', 4, 7)}")
    print(f"   potencia(2, 8) = {tools.ejecutar('potencia', 2, 8)}")
    print(f"   raiz(16, 2) = {tools.ejecutar('raiz', 16, 2)}")
    
    print("\n✅ HERRAMIENTAS FUNCIONANDO!")

if __name__ == "__main__":
    prueba_herramientas()
