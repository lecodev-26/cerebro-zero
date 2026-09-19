import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import urllib.request
import random

class ClimaReal:
    def __init__(self, api_key=None):
        # 🔑 API KEY REAL DE OPENWEATHERMAP
        self.api_key = api_key or "16a8ffb32af3e1ff123adb2ad4f71ad3"
        self.clima_cache = {}
        self.modo_real = self.api_key != "demo"
    
    def obtener_clima(self, ciudad="Madrid"):
        ciudad_lower = ciudad.lower()
        if self.modo_real:
            try:
                resultado = self._obtener_api(ciudad)
                if "Error" not in resultado:
                    return resultado
            except Exception as e:
                print(f"⚠️ API error: {e}, usando modo simulado")
        return self._obtener_simulado(ciudad_lower)
    
    def _obtener_api(self, ciudad):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={self.api_key}&units=metric&lang=es"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            humedad = data['main']['humidity']
            return f"🌡️ Clima en {ciudad}: {temp:.1f}°C, {desc}, humedad {humedad}%"
    
    def _obtener_simulado(self, ciudad):
        temperaturas = {
            'madrid': 28, 'barcelona': 26, 'valencia': 30, 'sevilla': 35,
            'bilbao': 22, 'malaga': 32, 'zaragoza': 33, 'granada': 34,
            'paris': 25, 'londres': 20, 'berlin': 22, 'roma': 29,
            'buenos aires': 32, 'mexico': 28, 'tokio': 30, 'new york': 24
        }
        condiciones = ['☀️ soleado', '⛅ nublado', '🌧️ lluvioso', '💨 ventoso', '☀️ despejado']
        temp = temperaturas.get(ciudad, random.randint(15, 35))
        cond = random.choice(condiciones)
        return f"🌡️ Clima en {ciudad.capitalize()}: {temp}°C, {cond} (simulado)"

if __name__ == "__main__":
    clima = ClimaReal()
    print(clima.obtener_clima('Madrid'))
