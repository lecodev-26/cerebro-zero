import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
import random
import json
import urllib.request
from datetime import datetime

class HerramientasExtra:
    def __init__(self):
        self.tools = {
            'clima': self.clima,
            'web': self.web_simulada,
            'calcular_edad': self.calcular_edad,
            'conversor_moneda': self.conversor_moneda,
            'dado': self.dado,
            'moneda': self.moneda,
            'numero_aleatorio': self.numero_aleatorio
        }
        self.clima_cache = {}
    
    def clima(self, ciudad="Madrid"):
        """Simula clima de una ciudad"""
        temp = random.randint(15, 35)
        condiciones = ['soleado', 'nublado', 'lluvioso', 'ventoso', 'nevando']
        cond = random.choice(condiciones)
        return f"Clima en {ciudad}: {temp}°C, {cond}"
    
    def web_simulada(self, url):
        """Simula scraping web"""
        return f"Contenido simulado de {url}: 'Página web de ejemplo con información interesante'"
    
    def calcular_edad(self, año, mes=1, dia=1):
        """Calcula la edad actual"""
        actual = datetime.now()
        try:
            nacimiento = datetime(año, mes, dia)
            edad = actual.year - nacimiento.year - ((actual.month, actual.day) < (nacimiento.month, nacimiento.day))
            return f"Edad: {edad} años"
        except:
            return "Error al calcular la edad"
    
    def conversor_moneda(self, cantidad, de_moneda="EUR", a_moneda="USD"):
        """Simula conversión de moneda"""
        tasas = {'EUR': 1.0, 'USD': 1.1, 'GBP': 0.85, 'JPY': 160.0, 'MXN': 20.0}
        if de_moneda in tasas and a_moneda in tasas:
            resultado = cantidad * tasas[a_moneda] / tasas[de_moneda]
            return f"{cantidad} {de_moneda} = {resultado:.2f} {a_moneda}"
        return f"Conversión no disponible para {de_moneda} → {a_moneda}"
    
    def dado(self, caras=6):
        """Lanza un dado"""
        return random.randint(1, caras)
    
    def moneda(self):
        """Lanza una moneda"""
        return random.choice(['cara', 'cruz'])
    
    def numero_aleatorio(self, min=1, max=100):
        """Genera un número aleatorio"""
        return random.randint(min, max)
    
    def ejecutar(self, comando, *args):
        if comando in self.tools:
            try:
                return self.tools[comando](*args)
            except Exception as e:
                return f"Error: {e}"
        return f"Herramienta desconocida: {comando}"
    
    def listar(self):
        return list(self.tools.keys())

def prueba_herramientas_extra():
    print("🧠 PROBANDO HERRAMIENTAS EXTRA")
    print("="*30)
    
    tools = HerramientasExtra()
    print(f"Herramientas: {tools.listar()}")
    
    print("\n🔧 Pruebas:")
    print(f"   clima('Barcelona') = {tools.ejecutar('clima', 'Barcelona')}")
    print(f"   web('https://ejemplo.com') = {tools.ejecutar('web', 'https://ejemplo.com')}")
    print(f"   calcular_edad(1990) = {tools.ejecutar('calcular_edad', 1990)}")
    print(f"   conversor_moneda(100, 'EUR', 'USD') = {tools.ejecutar('conversor_moneda', 100, 'EUR', 'USD')}")
    print(f"   dado(6) = {tools.ejecutar('dado', 6)}")
    print(f"   moneda() = {tools.ejecutar('moneda')}")
    print(f"   numero_aleatorio(1, 100) = {tools.ejecutar('numero_aleatorio', 1, 100)}")
    
    print("\n✅ HERRAMIENTAS EXTRA FUNCIONANDO!")

if __name__ == "__main__":
    prueba_herramientas_extra()
