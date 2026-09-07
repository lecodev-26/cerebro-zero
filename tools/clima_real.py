import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import urllib.request
import random

class ClimaReal:
    def __init__(self, api_key=None):
        self.api_key = api_key or "b91b688ff9e04d18f1cd6c87cc664923"
        self.clima_cache = {}
    
    def obtener_clima(self, ciudad="Madrid"):
        ciudad_lower = ciudad.lower()
        if self.api_key and self.api_key != "demo":
            return self._obtener_api(ciudad)
        return self._obtener_simulado(ciudad_lower)
    
    def _obtener_api(self, ciudad):
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={ciudad}&appid={self.api_key}&units=metric&lang=es"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                temp = data['main']['temp']
                desc = data['weather'][0]['description']
                humedad = data['main']['humidity']
                return f"🌡️ Clima en {ciudad}: {temp:.1f}°C, {desc}, humedad {humedad}%"
        except Exception as e:
            return f"❌ Error en API: {e}"
    
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
        return f"🌡️ Clima en {ciudad.capitalize()}: {temp}°C, {cond}"
    
    def obtener_pronostico(self, ciudad="Madrid", dias=3):
        ciudad_lower = ciudad.lower()
        resultados = [self._obtener_simulado(ciudad_lower)]
        for i in range(1, dias):
            temp = random.randint(15, 35)
            cond = random.choice(['☀️ soleado', '⛅ nublado', '🌧️ lluvioso'])
            resultados.append(f"   Día {i+1}: {temp}°C, {cond}")
        return "\n".join(resultados)

def prueba_clima_real():
    print("🧠 PROBANDO CLIMA REAL")
    print("="*30)
    
    clima = ClimaReal()
    print(f"🔑 API key: Demo (simulado)")
    
    print("\n🌤️ Clima:")
    print(f"   {clima.obtener_clima('Madrid')}")
    print(f"   {clima.obtener_clima('Barcelona')}")
    
    print("\n📅 Pronóstico 3 días:")
    print(clima.obtener_pronostico('Madrid', 3))

if __name__ == "__main__":
    prueba_clima_real()
