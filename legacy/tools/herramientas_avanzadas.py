import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
import random
from datetime import datetime

class HerramientasAvanzadas:
    def __init__(self):
        self.tools = {
            'sumar': self.sumar,
            'restar': self.restar,
            'multiplicar': self.multiplicar,
            'dividir': self.dividir,
            'potencia': self.potencia,
            'raiz': self.raiz,
            'contar': self.contar,
            'hora': self.hora,
            'fecha': self.fecha,
            'buscar': self.buscar_simulado,
            'traducir': self.traducir_simulado,
            'resumir': self.resumir_simulado,
            'memorizar': self.memorizar,
            'recordar': self.recordar
        }
        self.memoria_extra = {}
        self.conocimiento_base = {
            'capital_espana': 'Madrid',
            'capital_francia': 'París',
            'capital_italia': 'Roma',
            'capital_alemania': 'Berlín',
            'capital_portugal': 'Lisboa',
            'capital_inglaterra': 'Londres'
        }
    
    def sumar(self, a, b): return a + b
    def restar(self, a, b): return a - b
    def multiplicar(self, a, b): return a * b
    def dividir(self, a, b): return "Error: División por cero" if b == 0 else a / b
    def potencia(self, a, b): return a ** b
    def raiz(self, a, n=2): return a ** (1/n)
    def contar(self, lista): return len(lista)
    
    def hora(self):
        return datetime.now().strftime("%H:%M:%S")
    
    def fecha(self):
        return datetime.now().strftime("%d/%m/%Y")
    
    def buscar_simulado(self, query):
        """Simula una búsqueda en conocimiento base"""
        query_lower = query.lower()
        for key, value in self.conocimiento_base.items():
            if key.replace('_', ' ') in query_lower:
                return f"{query} → {value}"
        return f"Búsqueda simulada para: {query}"
    
    def traducir_simulado(self, texto, idioma):
        """Simula una traducción"""
        traducciones = {
            'hola': {'ingles': 'hello', 'frances': 'bonjour', 'aleman': 'hallo'},
            'adios': {'ingles': 'goodbye', 'frances': 'au revoir', 'aleman': 'auf wiedersehen'},
            'gracias': {'ingles': 'thank you', 'frances': 'merci', 'aleman': 'danke'}
        }
        if texto in traducciones and idioma in traducciones[texto]:
            return traducciones[texto][idioma]
        return f"Traducción simulada: {texto} → {idioma}"
    
    def resumir_simulado(self, texto):
        """Simula un resumen de texto"""
        palabras = texto.split()
        if len(palabras) <= 10:
            return texto
        return f"Resumen: {texto[:50]}..."
    
    def memorizar(self, clave, valor):
        """Guarda información en memoria extra"""
        self.memoria_extra[clave] = valor
        return f"Memorizado: {clave} = {valor}"
    
    def recordar(self, clave):
        """Recupera información de memoria extra"""
        return self.memoria_extra.get(clave, f"No tengo información sobre {clave}")
    
    def ejecutar(self, comando, *args):
        if comando in self.tools:
            try:
                return self.tools[comando](*args)
            except Exception as e:
                return f"Error: {e}"
        return f"Herramienta desconocida: {comando}"
    
    def listar(self):
        return list(self.tools.keys())
    
    def describir(self, comando):
        descripciones = {
            'sumar': 'Suma dos números',
            'restar': 'Resta dos números',
            'multiplicar': 'Multiplica dos números',
            'dividir': 'Divide dos números',
            'potencia': 'Eleva a a la b',
            'raiz': 'Calcula la raíz n-ésima',
            'contar': 'Cuenta elementos',
            'hora': 'Devuelve la hora actual',
            'fecha': 'Devuelve la fecha actual',
            'buscar': 'Busca información',
            'traducir': 'Traduce palabras',
            'resumir': 'Resume texto',
            'memorizar': 'Guarda información',
            'recordar': 'Recupera información'
        }
        return descripciones.get(comando, "Herramienta no descrita")

def prueba_herramientas_avanzadas():
    print("🧠 PROBANDO HERRAMIENTAS AVANZADAS")
    print("="*30)
    
    tools = HerramientasAvanzadas()
    print(f"Herramientas: {tools.listar()}")
    
    print("\n🔧 Pruebas:")
    print(f"   hora() = {tools.ejecutar('hora')}")
    print(f"   fecha() = {tools.ejecutar('fecha')}")
    print(f"   buscar('capital de España') = {tools.ejecutar('buscar', 'capital de España')}")
    print(f"   traducir('hola', 'ingles') = {tools.ejecutar('traducir', 'hola', 'ingles')}")
    print(f"   memorizar('mi_color', 'azul') = {tools.ejecutar('memorizar', 'mi_color', 'azul')}")
    print(f"   recordar('mi_color') = {tools.ejecutar('recordar', 'mi_color')}")
    
    print("\n✅ HERRAMIENTAS AVANZADAS FUNCIONANDO!")

if __name__ == "__main__":
    prueba_herramientas_avanzadas()
