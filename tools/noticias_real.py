import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import urllib.request
import random

class NoticiasReal:
    def __init__(self, api_key=None):
        self.api_key = api_key or "2f6c9736507e4499bcd66c52858ffc58"
        self.categorias = {
            'general': 'general',
            'tecnologia': 'technology',
            'deportes': 'sports',
            'ciencia': 'science',
            'economia': 'business',
            'entretenimiento': 'entertainment',
            'salud': 'health'
        }
    
    def obtener_noticias(self, categoria="general", cantidad=5):
        """Obtiene noticias reales (simulado sin API key)"""
        if self.api_key and self.api_key != "demo":
            return self._obtener_api(categoria, cantidad)
        return self._obtener_simulado(categoria, cantidad)
    
    def _obtener_api(self, categoria, cantidad):
        """Llama a la API de NewsAPI"""
        try:
            cat = self.categorias.get(categoria, 'general')
            url = f"https://newsapi.org/v2/top-headlines?category={cat}&language=es&pageSize={cantidad}&apiKey={self.api_key}"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())
                if data['status'] == 'ok':
                    noticias = []
                    for article in data['articles']:
                        titulo = article.get('title', 'Sin título')
                        fuente = article.get('source', {}).get('name', '')
                        noticias.append(f"📰 {titulo} ({fuente})")
                    return "\n".join(noticias[:cantidad])
                return f"❌ Error en API: {data.get('message', 'Desconocido')}"
        except Exception as e:
            return f"❌ Error en API: {e}"
    
    def _obtener_simulado(self, categoria, cantidad):
        """Modo simulado sin API"""
        noticias = {
            'general': [
                "Nuevo avance en IA desde España",
                "El mercado tecnológico crece un 20%",
                "Innovación en robótica revoluciona la industria",
                "Descubren nuevo planeta en el sistema solar",
                "La inteligencia artificial llega a los hogares"
            ],
            'tecnologia': [
                "Internet 6G en desarrollo para 2030",
                "Revolución en baterías de estado sólido",
                "Lanzamiento del nuevo smartphone plegable",
                "Los coches autónomos ya son realidad",
                "La computación cuántica da su primer gran paso"
            ],
            'deportes': [
                "El Real Madrid gana la Champions League",
                "Messi anuncia su retiro del fútbol",
                "Nuevo récord mundial en natación",
                "España gana el Mundial de Fútbol",
                "La selección de baloncesto hace historia"
            ],
            'ciencia': [
                "Vacuna contra el cáncer en fase final",
                "Inteligencia artificial cuántica revoluciona la física",
                "Descubren vida en Marte",
                "La fusión nuclear da su primer paso comercial",
                "Nuevo telescopio revela los secretos del universo"
            ],
            'economia': [
                "La inflación baja al 2% en Europa",
                "Bitcoin alcanza nuevo máximo histórico",
                "Nuevo tratado comercial entre la UE y China",
                "La bolsa cierra al alza",
                "El paro baja al nivel más bajo en 10 años"
            ]
        }
        
        noticias_cat = noticias.get(categoria, noticias['general'])
        seleccionadas = random.sample(noticias_cat, min(cantidad, len(noticias_cat)))
        return "\n".join([f"📰 {n}" for n in seleccionadas])
    
    def buscar_noticias(self, query, cantidad=5):
        """Busca noticias por palabra clave"""
        if self.api_key and self.api_key != "demo":
            try:
                url = f"https://newsapi.org/v2/everything?q={query}&language=es&pageSize={cantidad}&apiKey={self.api_key}"
                with urllib.request.urlopen(url, timeout=5) as response:
                    data = json.loads(response.read())
                    if data['status'] == 'ok':
                        noticias = []
                        for article in data['articles']:
                            titulo = article.get('title', 'Sin título')
                            fuente = article.get('source', {}).get('name', '')
                            noticias.append(f"🔍 {titulo} ({fuente})")
                        return "\n".join(noticias[:cantidad])
            except:
                pass
        return self._obtener_simulado('general', cantidad)

def prueba_noticias_real():
    print("🧠 PROBANDO NOTICIAS REALES")
    print("="*30)
    
    noticias = NoticiasReal()
    print(f"🔑 API key: {'Configurada' if noticias.api_key != 'demo' else 'Demo (simulado)'}")
    
    print("\n📰 Noticias de Tecnología:")
    print(noticias.obtener_noticias('tecnologia', 3))
    
    print("\n📰 Noticias de Deportes:")
    print(noticias.obtener_noticias('deportes', 3))

if __name__ == "__main__":
    prueba_noticias_real()
