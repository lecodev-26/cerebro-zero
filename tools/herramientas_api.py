import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
import json
import urllib.request
import random
from datetime import datetime

class HerramientasAPI:
    def __init__(self):
        self.tools = {
            'noticias': self.noticias,
            'clima_real': self.clima_real,
            'chiste': self.chiste,
            'cita': self.cita,
            'definicion': self.definicion,
            'edad': self.calcular_edad,
            'conversor': self.conversor_moneda,
            'recordar': self.recordar,
            'memorizar': self.memorizar
        }
        self.memoria = {}
        self.citas = [
            "La inteligencia artificial no es una amenaza, es una oportunidad.",
            "El futuro pertenece a quienes creen en la belleza de sus sueños.",
            "La única forma de hacer un gran trabajo es amar lo que haces.",
            "La creatividad es la inteligencia divirtiéndose.",
            "El aprendizaje nunca agota la mente."
        ]
        self.chistes = [
            "¿Qué le dice un bit a otro? ¡Nos vemos en el bus!",
            "¿Cómo se llama el campeón de buceo japonés? Tokofondo.",
            "¿Qué hace una abeja en el gimnasio? ¡Zum-ba!",
            "¿Cómo se dice 'aparcamiento' en chino? ¡Párking!",
            "¿Qué le dice un semáforo a otro? No me mires, que me pongo rojo."
        ]
    
    def noticias(self, categoria="general"):
        """Simula noticias de una categoría"""
        noticias = {
            'general': ["Nuevo avance en IA desde España", "El mercado tecnológico crece un 20%", "Innovación en robótica"],
            'deportes': ["El Real Madrid gana la Champions", "Messi anuncia su retiro", "Nuevo récord en natación"],
            'ciencia': ["Descubren nuevo planeta", "Vacuna contra el cáncer en fase final", "Inteligencia artificial cuántica"],
            'tecnologia': ["Lanzamiento de nuevo smartphone", "Internet 6G en desarrollo", "Revolución en baterías"]
        }
        return random.choice(noticias.get(categoria, noticias['general']))
    
    def clima_real(self, ciudad="Madrid"):
        """Simula clima real (sin API externa)"""
        temperaturas = {
            'madrid': 28, 'barcelona': 26, 'valencia': 30, 'sevilla': 35, 
            'bilbao': 22, 'malaga': 32, 'zaragoza': 33, 'granada': 34
        }
        condiciones = ['soleado', 'nublado', 'lluvioso', 'ventoso', 'despejado']
        ciudad_lower = ciudad.lower()
        
        temp = temperaturas.get(ciudad_lower, random.randint(15, 35))
        cond = random.choice(condiciones)
        return f"🌡️ Clima en {ciudad}: {temp}°C, {cond}"
    
    def chiste(self):
        """Devuelve un chiste aleatorio"""
        return f"😂 {random.choice(self.chistes)}"
    
    def cita(self):
        """Devuelve una cita inspiradora"""
        return f"💡 '{random.choice(self.citas)}'"
    
    def definicion(self, palabra):
        """Simula una definición de palabra"""
        definiciones = {
            'ia': 'Inteligencia Artificial: capacidad de una máquina para imitar funciones humanas.',
            'ia': 'Inteligencia Artificial: sistema que aprende y toma decisiones.',
            'red': 'Red neuronal: sistema computacional inspirado en el cerebro humano.',
            'algoritmo': 'Algoritmo: secuencia de pasos para resolver un problema.',
            'datos': 'Datos: información organizada que sirve como base para el aprendizaje.'
        }
        palabra_lower = palabra.lower()
        return definiciones.get(palabra_lower, f"No tengo definición para '{palabra}'")
    
    def calcular_edad(self, año, mes=1, dia=1):
        try:
            actual = datetime.now()
            nacimiento = datetime(año, mes, dia)
            edad = actual.year - nacimiento.year - ((actual.month, actual.day) < (nacimiento.month, nacimiento.day))
            return f"📅 Edad: {edad} años"
        except:
            return "Error al calcular la edad"
    
    def conversor_moneda(self, cantidad, de_moneda="EUR", a_moneda="USD"):
        tasas = {'EUR': 1.0, 'USD': 1.1, 'GBP': 0.85, 'JPY': 160.0, 'MXN': 20.0, 'ARS': 950.0}
        if de_moneda in tasas and a_moneda in tasas:
            resultado = cantidad * tasas[a_moneda] / tasas[de_moneda]
            return f"💱 {cantidad} {de_moneda} = {resultado:.2f} {a_moneda}"
        return f"Conversión no disponible para {de_moneda} → {a_moneda}"
    
    def memorizar(self, clave, valor):
        self.memoria[clave] = valor
        return f"🧠 Memorizado: {clave} = {valor}"
    
    def recordar(self, clave):
        return self.memoria.get(clave, f"No tengo información sobre '{clave}'")
    
    def ejecutar(self, comando, *args):
        if comando in self.tools:
            try:
                return self.tools[comando](*args)
            except Exception as e:
                return f"Error: {e}"
        return f"Herramienta desconocida: {comando}"
    
    def listar(self):
        return list(self.tools.keys())

def prueba_herramientas_api():
    print("🧠 PROBANDO HERRAMIENTAS API")
    print("="*30)
    
    tools = HerramientasAPI()
    print(f"Herramientas: {tools.listar()}")
    
    print("\n🔧 Pruebas:")
    print(f"   noticias('tecnologia') = {tools.ejecutar('noticias', 'tecnologia')}")
    print(f"   clima_real('Madrid') = {tools.ejecutar('clima_real', 'Madrid')}")
    print(f"   chiste() = {tools.ejecutar('chiste')}")
    print(f"   cita() = {tools.ejecutar('cita')}")
    print(f"   definicion('ia') = {tools.ejecutar('definicion', 'ia')}")
    print(f"   conversor(100, 'EUR', 'USD') = {tools.ejecutar('conversor', 100, 'EUR', 'USD')}")
    
    print("\n✅ HERRAMIENTAS API FUNCIONANDO!")

if __name__ == "__main__":
    prueba_herramientas_api()
