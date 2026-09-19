import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import urllib.request
import random

class NoticiasReal:
    def __init__(self, api_key=None):
        # 🔑 API KEY REAL DE NEWSAPI
        self.api_key = api_key or "2f6c9736507e4499bcd66c52858ffc58"
        self.modo_real = self.api_key != "demo"
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
        cat = self.categorias.get(categoria, 'general')
        if self.modo_real:
            try:
                resultado = self._obtener_api(cat, cantidad)
                if "Error" not in resultado:
                    return resultado
            except Exception as e:
                print(f"⚠️ API error: {e}, usando modo simulado")
        return self._obtener_simulado(categoria, cantidad)
    
    def _obtener_api(self, categoria, cantidad):
        url = f"https://newsapi.org/v2/top-headlines?category={categoria}&language=es&pageSize={cantidad}&apiKey={self.api_key}"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
            if data['status'] == 'ok':
                noticias = []
                for article in data['articles']:
                    titulo = article.get('title', 'Sin título')
                    fuente = article.get('source', {}).get('name', '')
                    noticias.append(f"📰 {titulo} ({fuente})")
                return "\n".join(noticias[:cantidad])
            return f"❌ Error: {data.get('message', 'Desconocido')}"
    
    def _obtener_simulado(self, categoria, cantidad):
        noticias = {
            'general': ["Nuevo avance en IA desde España", "El mercado tecnológico crece un 20%"],
            'tecnologia': ["Internet 6G en desarrollo", "Revolución en baterías"],
            'deportes': ["El Real Madrid gana la Champions", "Messi anuncia su retiro"]
        }
        noticias_cat = noticias.get(categoria, noticias['general'])
        seleccionadas = random.sample(noticias_cat, min(cantidad, len(noticias_cat)))
        return "\n".join([f"📰 {n}" for n in seleccionadas])
    
    def buscar_noticias(self, query, cantidad=5):
        if self.modo_real:
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

if __name__ == "__main__":
    noticias = NoticiasReal()
    print(noticias.obtener_noticias('tecnologia', 2))
