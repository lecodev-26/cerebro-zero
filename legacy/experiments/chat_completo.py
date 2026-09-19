import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
from memory.memory import MemoriaPersistente
from tools.herramientas_avanzadas import HerramientasAvanzadas
from tools.herramientas_extra import HerramientasExtra
from tools.herramientas_api import HerramientasAPI
from tools.herramientas_reales import HerramientasReales
from tools.voz import ReconocimientoVoz

class ChatCompleto:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="chat_completo.json", max_size=100)
        self.herramientas = HerramientasAvanzadas()
        self.herramientas_extra = HerramientasExtra()
        self.herramientas_api = HerramientasAPI()
        self.herramientas_reales = HerramientasReales()
        self.voz = ReconocimientoVoz()
        self.nombre = "Cerebro Zero"
        self.historial = []
    
    def procesar(self, mensaje):
        mensaje = mensaje.strip()
        
        if mensaje.startswith('/'):
            return self._procesar_comando(mensaje)
        
        if mensaje.lower() in ['escuchar', 'voz', 'micrófono']:
            return self._escuchar_voz()
        
        resultado = self._detectar_herramienta(mensaje)
        if resultado:
            return resultado
        
        self.memoria.aprender([len(mensaje)], [len(mensaje) % 2])
        self.historial.append(mensaje)
        
        return self._respuesta_generica(mensaje)
    
    def _detectar_herramienta(self, mensaje):
        mensaje_lower = mensaje.lower()
        
        # Horóscopo
        if 'horoscopo' in mensaje_lower:
            palabras = mensaje.split()
            signo = palabras[1] if len(palabras) > 1 else 'general'
            return self.herramientas_reales.ejecutar('horoscopo', signo)
        
        # Fecha
        if mensaje_lower in ['fecha', 'fecha actual']:
            return self.herramientas_reales.ejecutar('fecha')
        
        # Hora
        if mensaje_lower in ['hora', 'hora actual']:
            return self.herramientas_reales.ejecutar('hora')
        
        # Consejo
        if 'consejo' in mensaje_lower:
            return self.herramientas_reales.ejecutar('consejo')
        
        # Tarjeta
        if 'tarjeta' in mensaje_lower:
            numeros = re.findall(r'\d+', mensaje)
            if numeros:
                return self.herramientas_reales.ejecutar('tarjeta', numeros[0])
            return "❌ Necesito un número de tarjeta"
        
        # Lista de tareas
        if 'lista' in mensaje_lower:
            partes = mensaje.split()
            if len(partes) >= 2:
                accion = partes[1]
                tarea = ' '.join(partes[2:]) if len(partes) > 2 else ''
                return self.herramientas_reales.ejecutar('lista_tareas', accion, tarea)
            return self.herramientas_reales.ejecutar('lista_tareas', 'ver')
        
        # Seguro
        if 'seguro' in mensaje_lower:
            numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
            if len(numeros) >= 2:
                return self.herramientas_reales.ejecutar('seguro', float(numeros[0]), int(numeros[1]))
            return "❌ Usa: seguro [monto] [años]"
        
        # Calculadora
        if 'calculadora' in mensaje_lower:
            palabras = mensaje.split()
            if len(palabras) >= 4:
                return self.herramientas_reales.ejecutar('calculadora', palabras[1], float(palabras[2]), float(palabras[3]))
            return "❌ Usa: calculadora [suma/resta/multiplica/divide] [a] [b]"
        
        # Noticias
        if 'noticias' in mensaje_lower:
            palabras = mensaje.split()
            categoria = palabras[1] if len(palabras) > 1 else 'general'
            return self.herramientas_api.ejecutar('noticias', categoria)
        
        # Clima
        if 'clima' in mensaje_lower:
            palabras = mensaje.split()
            ciudad = ' '.join(palabras[1:]) if len(palabras) > 1 else 'Madrid'
            return self.herramientas_reales.ejecutar('clima', ciudad)
        
        # Chiste
        if 'chiste' in mensaje_lower:
            return self.herramientas_api.ejecutar('chiste')
        
        # Cita
        if 'cita' in mensaje_lower:
            return self.herramientas_api.ejecutar('cita')
        
        # Definición
        if 'definicion' in mensaje_lower or 'definir' in mensaje_lower:
            palabra = mensaje.replace('definicion', '').replace('definir', '').strip()
            if palabra:
                return self.herramientas_api.ejecutar('definicion', palabra)
            return "❌ ¿Qué palabra quieres definir?"
        
        # Conversor
        if 'conversor' in mensaje_lower or 'conversión' in mensaje_lower:
            numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
            if numeros:
                return self.herramientas_api.ejecutar('conversor', float(numeros[0]), 'EUR', 'USD')
            return "❌ Usa: conversor [cantidad]"
        
        # Traducir
        if 'traducir' in mensaje_lower:
            partes = mensaje.replace('traducir', '').strip().split()
            if len(partes) >= 2:
                return self.herramientas.ejecutar('traducir', partes[0], partes[1] if len(partes) > 1 else 'ingles')
            return "❌ Usa: traducir [palabra] [idioma]"
        
        # Dado
        if 'dado' in mensaje_lower:
            numeros = re.findall(r'\d+', mensaje)
            caras = int(numeros[0]) if numeros else 6
            return self.herramientas_extra.ejecutar('dado', caras)
        
        # Moneda
        if 'moneda' in mensaje_lower:
            return self.herramientas_extra.ejecutar('moneda')
        
        # Memorizar
        if 'memorizar' in mensaje_lower:
            partes = mensaje.split('memorizar', 1)[1].strip().split('=', 1)
            if len(partes) == 2:
                return self.herramientas.ejecutar('memorizar', partes[0].strip(), partes[1].strip())
            return "❌ Usa: memorizar [clave] = [valor]"
        
        # Recordar
        if 'recordar' in mensaje_lower:
            clave = mensaje.replace('recordar', '').strip()
            if clave:
                return self.herramientas.ejecutar('recordar', clave)
            return "❌ ¿Qué quieres recordar?"
        
        return None
    
    def _escuchar_voz(self):
        texto = self.voz.escuchar(duracion=3)
        if texto:
            return f"🎤 He escuchado: '{texto}'. {self.procesar(texto)}"
        return "🎤 No he podido escuchar nada."
    
    def _procesar_comando(self, mensaje):
        comando = mensaje[1:].split()[0] if len(mensaje) > 1 else ''
        
        if comando == 'ayuda':
            return "📖 Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /voz"
        elif comando == 'historial':
            return f"📝 Últimos mensajes: {self.historial[-5:]}"
        elif comando == 'olvidar':
            self.memoria.olvidar()
            return "🧠 Memoria borrada"
        elif comando == 'nombre':
            return f"🧠 Me llamo {self.nombre}"
        elif comando == 'herramientas':
            return f"🔧 Herramientas: {list(self.herramientas_reales.listar() + self.herramientas_api.listar() + self.herramientas_extra.listar())}"
        elif comando == 'voz':
            return self._escuchar_voz()
        else:
            return f"❌ Comando desconocido: {comando}"
    
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
        print("   - 'horoscopo leo' → Horóscopo")
        print("   - 'fecha' → Fecha actual")
        print("   - 'hora' → Hora actual")
        print("   - 'consejo' → Consejo")
        print("   - 'tarjeta 1234567890123456' → Validar tarjeta")
        print("   - 'lista añadir Tarea' → Añadir tarea")
        print("   - 'lista ver' → Ver tareas")
        print("   - 'seguro 10000 10' → Seguro de vida")
        print("   - 'calculadora suma 5 3' → Calculadora")
        print("   - 'noticias tecnologia' → Noticias")
        print("   - 'clima Madrid' → Clima")
        print("   - 'chiste' → Chiste")
        print("   - 'cita' → Cita")
        print("   - 'definicion ia' → Definición")
        print("   - 'traducir hola ingles' → Traducción")
        print("   - 'dado 6' → Lanzar dado")
        print("   - 'memorizar color = azul' → Memorizar")
        print("   - 'recordar color' → Recordar")
        print("   - 'voz' o 'escuchar' → Voz")
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
