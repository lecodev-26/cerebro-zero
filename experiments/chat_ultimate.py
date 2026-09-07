import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import random
from memory.memory import MemoriaPersistente
from tools.herramientas_reales import HerramientasReales
from tools.voz_real import VozReal
from tools.clima_real import ClimaReal
from tools.noticias_real import NoticiasReal

class ChatUltimate:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="chat_ultimate.json", max_size=100)
        self.herramientas = HerramientasReales()
        self.voz = VozReal()
        self.clima = ClimaReal()
        self.noticias = NoticiasReal()
        self.nombre = "Cerebro Zero Ultimate"
        self.historial = []
    
    def procesar(self, mensaje):
        mensaje = mensaje.strip()
        
        if mensaje.startswith('/'):
            return self._comandos(mensaje)
        
        if mensaje.lower() in ['escuchar', 'voz', 'micrófono']:
            return self._escuchar_voz()
        
        resultado = self._herramientas(mensaje)
        if resultado:
            return resultado
        
        self.memoria.aprender([len(mensaje)], [len(mensaje) % 2])
        self.historial.append(mensaje)
        return self._respuesta(mensaje)
    
    def _herramientas(self, mensaje):
        m = mensaje.lower()
        
        if 'clima' in m or 'temperatura' in m:
            ciudad = re.sub(r'(clima|temperatura|en|de)\s*', '', mensaje).strip()
            ciudad = ciudad if ciudad else 'Madrid'
            return self.clima.obtener_clima(ciudad)
        
        if 'noticias' in m:
            if 'buscar' in m:
                query = m.replace('buscar noticias', '').replace('buscar', '').replace('noticias', '').strip()
                if query:
                    return self.noticias.buscar_noticias(query, 3)
                return "¿Qué quieres buscar?"
            cat = 'tecnologia' if 'tecno' in m else 'deportes' if 'deporte' in m else 'general'
            return self.noticias.obtener_noticias(cat, 3)
        
        if 'horoscopo' in m:
            signo = re.sub(r'horoscopo\s*', '', mensaje).strip()
            signo = signo if signo else 'general'
            return self.herramientas.ejecutar('horoscopo', signo)
        
        if 'fecha' in m and 'hora' not in m:
            return self.herramientas.ejecutar('fecha')
        
        if 'hora' in m:
            return self.herramientas.ejecutar('hora')
        
        if 'consejo' in m:
            return self.herramientas.ejecutar('consejo')
        
        if 'chiste' in m:
            return self.herramientas.ejecutar('chiste')
        
        if 'cita' in m:
            return self.herramientas.ejecutar('cita')
        
        if 'definicion' in m or 'definir' in m:
            palabra = re.sub(r'(definicion|definir|de)\s*', '', mensaje).strip()
            if palabra:
                return self.herramientas.ejecutar('definicion', palabra)
            return "¿Qué palabra quieres definir?"
        
        if 'conversor' in m:
            numeros = re.findall(r'\d+', m)
            if numeros:
                return self.herramientas.ejecutar('conversor', float(numeros[0]), 'EUR', 'USD')
            return "¿Qué cantidad quieres convertir?"
        
        if 'dado' in m:
            numeros = re.findall(r'\d+', m)
            caras = int(numeros[0]) if numeros else 6
            return self.herramientas.ejecutar('dado', caras)
        
        if 'memorizar' in m:
            if '=' in m:
                partes = m.split('memorizar', 1)[1].strip().split('=', 1)
                if len(partes) == 2:
                    return self.herramientas.ejecutar('memorizar', partes[0].strip(), partes[1].strip())
            return "Usa: memorizar [clave] = [valor]"
        
        if 'recordar' in m:
            clave = m.replace('recordar', '').strip()
            if clave:
                return self.herramientas.ejecutar('recordar', clave)
            return "¿Qué quieres recordar?"
        
        return None
    
    def _escuchar_voz(self):
        texto = self.voz.escuchar(3)
        if texto:
            return f"🎤 He escuchado: '{texto}'. {self.procesar(texto)}"
        return "🎤 No he podido escuchar nada. Inténtalo de nuevo."
    
    def _comandos(self, mensaje):
        comando = mensaje[1:].split()[0] if len(mensaje) > 1 else ''
        
        if comando == 'ayuda':
            return "📖 /ayuda, /historial, /olvidar, /nombre, /herramientas, /voz"
        elif comando == 'historial':
            return f"📝 Últimos mensajes: {self.historial[-5:]}"
        elif comando == 'olvidar':
            self.memoria.olvidar()
            return "🧠 Memoria borrada"
        elif comando == 'nombre':
            return f"🧠 Me llamo {self.nombre}"
        elif comando == 'herramientas':
            return f"🔧 Herramientas: {self.herramientas.listar()}"
        elif comando == 'voz':
            return self._escuchar_voz()
        else:
            return f"❌ Comando desconocido: {comando}"
    
    def _respuesta(self, mensaje):
        respuestas = [
            "Interesante. ¿Qué más?",
            "No estoy seguro de entender eso.",
            "¿Podrías reformularlo?",
            "Dime más sobre eso.",
            "Entendido. ¿Algo más?"
        ]
        return random.choice(respuestas)
    
    def chat(self):
        print(f"\n🧠 {self.nombre}")
        print("="*50)
        print("💡 EJEMPLOS:")
        print("   'clima Madrid' → Clima")
        print("   'noticias tecnologia' → Noticias")
        print("   'buscar noticias inteligencia artificial' → Búsqueda")
        print("   'chiste' → Chiste")
        print("   'cita' → Cita inspiradora")
        print("   'horoscopo leo' → Horóscopo")
        print("   'fecha' → Fecha actual")
        print("   'hora' → Hora actual")
        print("   'consejo' → Consejo")
        print("   'definicion ia' → Definición")
        print("   'conversor 100' → Conversor EUR a USD")
        print("   'dado 6' → Lanzar dado")
        print("   'memorizar color = azul' → Memorizar")
        print("   'recordar color' → Recordar")
        print("   'voz' o 'escuchar' → Voz")
        print("="*50)
        print("Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /voz")
        print("Escribe 'salir' para terminar")
        print("="*50)
        
        while True:
            try:
                mensaje = input("\n🧑 Tú: ")
                if mensaje.lower() in ['salir', 'exit']:
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
    chat = ChatUltimate()
    chat.chat()
