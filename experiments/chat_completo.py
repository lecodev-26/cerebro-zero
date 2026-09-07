import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
from memory.memory import MemoriaPersistente
from tools.herramientas_avanzadas import HerramientasAvanzadas
from tools.herramientas_extra import HerramientasExtra
from tools.herramientas_api import HerramientasAPI
from tools.voz import ReconocimientoVoz

class ChatCompleto:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="chat_completo.json", max_size=100)
        self.herramientas = HerramientasAvanzadas()
        self.herramientas_extra = HerramientasExtra()
        self.herramientas_api = HerramientasAPI()
        self.voz = ReconocimientoVoz()
        self.nombre = "Cerebro Zero"
        self.historial = []
    
    def procesar(self, mensaje):
        mensaje = mensaje.strip()
        
        # Comandos especiales
        if mensaje.startswith('/'):
            return self._procesar_comando(mensaje)
        
        # Escucha por voz
        if mensaje.lower() in ['escuchar', 'voz', 'micrófono']:
            return self._escuchar_voz()
        
        # Intentar herramientas API primero
        resultado = self._detectar_herramienta_api(mensaje)
        if resultado:
            return resultado
        
        # Intentar herramientas avanzadas
        resultado = self._detectar_herramienta_avanzada(mensaje)
        if resultado:
            return resultado
        
        # Intentar herramientas extra
        resultado = self._detectar_herramienta_extra(mensaje)
        if resultado:
            return resultado
        
        # Guardar en memoria
        self.memoria.aprender([len(mensaje)], [len(mensaje) % 2])
        self.historial.append(mensaje)
        
        return self._respuesta_generica(mensaje)
    
    def _escuchar_voz(self):
        texto = self.voz.escuchar(duracion=3)
        if texto:
            return f"🎤 He escuchado: '{texto}'. {self.procesar(texto)}"
        return "🎤 No he podido escuchar nada. Inténtalo de nuevo."
    
    def _detectar_herramienta_api(self, mensaje):
        mensaje_lower = mensaje.lower()
        herramientas = self.herramientas_api.listar()
        
        for tool in herramientas:
            if tool in mensaje_lower:
                if tool == 'noticias':
                    categoria = mensaje.replace('noticias', '').strip() or 'general'
                    return f"📰 {self.herramientas_api.ejecutar('noticias', categoria)}"
                elif tool == 'clima_real':
                    ciudad = mensaje.replace('clima_real', '').replace('clima', '').strip() or 'Madrid'
                    return self.herramientas_api.ejecutar('clima_real', ciudad)
                elif tool == 'chiste':
                    return self.herramientas_api.ejecutar('chiste')
                elif tool == 'cita':
                    return self.herramientas_api.ejecutar('cita')
                elif tool == 'definicion':
                    palabra = mensaje.replace('definicion', '').replace('definir', '').strip()
                    if palabra:
                        return self.herramientas_api.ejecutar('definicion', palabra)
                    return "¿Qué palabra quieres definir?"
                elif tool == 'conversor':
                    numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
                    if numeros:
                        return self.herramientas_api.ejecutar('conversor', float(numeros[0]), 'EUR', 'USD')
        return None
    
    def _detectar_herramienta_avanzada(self, mensaje):
        numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
        if 'sumar' in mensaje and len(numeros) >= 2:
            return f"{numeros[0]} + {numeros[1]} = {self.herramientas.ejecutar('sumar', float(numeros[0]), float(numeros[1]))}"
        if 'traducir' in mensaje:
            partes = mensaje.replace('traducir', '').strip().split()
            if len(partes) >= 2:
                return self.herramientas.ejecutar('traducir', partes[0], partes[1] if len(partes) > 1 else 'ingles')
        if 'memorizar' in mensaje:
            partes = mensaje.split('memorizar', 1)[1].strip().split('=', 1)
            if len(partes) == 2:
                return self.herramientas.ejecutar('memorizar', partes[0].strip(), partes[1].strip())
        if 'recordar' in mensaje:
            clave = mensaje.replace('recordar', '').strip()
            return self.herramientas.ejecutar('recordar', clave)
        return None
    
    def _detectar_herramienta_extra(self, mensaje):
        if 'dado' in mensaje:
            numeros = re.findall(r'\d+', mensaje)
            caras = int(numeros[0]) if numeros else 6
            return f"🎲 Dado: {self.herramientas_extra.ejecutar('dado', caras)}"
        if 'moneda' in mensaje:
            return f"🪙 {self.herramientas_extra.ejecutar('moneda')}"
        return None
    
    def _procesar_comando(self, mensaje):
        partes = mensaje.split()
        comando = partes[0][1:]
        
        comandos = {
            'ayuda': "Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /voz",
            'historial': f"Últimos mensajes: {self.historial[-5:]}",
            'olvidar': self._olvidar_memoria(),
            'nombre': f"Me llamo {self.nombre}",
            'herramientas': f"Herramientas: {self.herramientas.listar() + self.herramientas_extra.listar() + self.herramientas_api.listar()}",
            'voz': self._escuchar_voz()
        }
        return comandos.get(comando, f"Comando desconocido: {comando}")
    
    def _olvidar_memoria(self):
        self.memoria.olvidar()
        return "🧠 Memoria borrada"
    
    def _respuesta_generica(self, mensaje):
        respuestas = [
            f"Entendido: {mensaje}",
            "Interesante. ¿Qué más?",
            "No estoy seguro de entender eso.",
            "¿Podrías reformularlo?",
            "Dime más sobre eso."
        ]
        return np.random.choice(respuestas)
    
    def chat(self):
        print(f"\n🧠 {self.nombre} - Chat Completo")
        print("="*50)
        print("💡 EJEMPLOS DE USO:")
        print("   - 'clima Madrid' → Clima")
        print("   - 'dado 6' → Lanzar dado")
        print("   - 'noticias tecnologia' → Noticias")
        print("   - 'chiste' → Chiste")
        print("   - 'cita' → Cita inspiradora")
        print("   - 'definicion ia' → Definición")
        print("   - 'traducir hola ingles' → Traducción")
        print("   - 'memorizar color = azul' → Memorizar")
        print("   - 'recordar color' → Recordar")
        print("   - 'voz' o 'escuchar' → Reconocimiento de voz")
        print("="*50)
        print("Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /voz")
        print("Escribe 'salir' para terminar")
        print("="*50)
        
        while True:
            try:
                mensaje = input("\n🧑 Tú: ")
                if mensaje.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                respuesta = self.procesar(mensaje)
                print(f"🧠 {self.nombre}: {respuesta}")
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat = ChatCompleto()
    chat.chat()
