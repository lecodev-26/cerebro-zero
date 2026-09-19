import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.memory import MemoriaPersistente
from tools.herramientas_avanzadas import HerramientasAvanzadas

class ChatInteractivo:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="chat_memoria.json", max_size=100)
        self.herramientas = HerramientasAvanzadas()
        self.nombre = "Cerebro Zero"
        self.historial = []
        self.recordar_ultimas = 5
    
    def procesar(self, mensaje):
        mensaje = mensaje.strip().lower()
        
        # Comandos especiales
        if mensaje.startswith('/'):
            return self._procesar_comando(mensaje)
        
        # Detectar si es una pregunta
        if mensaje.endswith('?'):
            return self._responder_pregunta(mensaje)
        
        # Intentar usar herramientas
        herramienta = self._detectar_herramienta(mensaje)
        if herramienta:
            return herramienta
        
        # Respuesta genérica
        return self._respuesta_generica(mensaje)
    
    def _procesar_comando(self, mensaje):
        partes = mensaje.split()
        comando = partes[0][1:]
        
        if comando == 'ayuda':
            return "Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas"
        elif comando == 'historial':
            return f"Últimos mensajes: {self.historial[-5:]}"
        elif comando == 'olvidar':
            self.memoria.olvidar()
            return "Memoria borrada"
        elif comando == 'nombre':
            return f"Me llamo {self.nombre}"
        elif comando == 'herramientas':
            return f"Herramientas disponibles: {self.herramientas.listar()}"
        else:
            return f"Comando desconocido: {comando}"
    
    def _detectar_herramienta(self, mensaje):
        # Detectar operaciones matemáticas simples
        import re
        patrones = [
            (r'suma|sumar|\+', 'sumar'),
            (r'resta|restar|-', 'restar'),
            (r'multiplica|multiplicar|\*', 'multiplicar'),
            (r'divide|dividir|/', 'dividir'),
            (r'potencia|elevar|\^', 'potencia'),
            (r'raiz|sqrt|√', 'raiz'),
            (r'contar|cuantos', 'contar'),
            (r'hora', 'hora'),
            (r'fecha', 'fecha'),
            (r'buscar|encontrar', 'buscar'),
            (r'traducir', 'traducir'),
            (r'resumir', 'resumir'),
            (r'memorizar|recordar', 'recordar')
        ]
        
        numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
        if numeros and len(numeros) >= 2:
            for patron, herramienta in patrones:
                if re.search(patron, mensaje):
                    if herramienta in ['sumar', 'restar', 'multiplicar', 'dividir', 'potencia']:
                        resultado = self.herramientas.ejecutar(herramienta, float(numeros[0]), float(numeros[1]))
                        return f"{herramienta}({numeros[0]}, {numeros[1]}) = {resultado}"
                    elif herramienta == 'raiz' and len(numeros) >= 1:
                        resultado = self.herramientas.ejecutar('raiz', float(numeros[0]))
                        return f"√({numeros[0]}) = {resultado:.4f}"
                    elif herramienta == 'contar':
                        return f"Hay {len(numeros)} números"
        
        # Otras herramientas
        if 'memorizar' in mensaje:
            partes = mensaje.split('memorizar', 1)[1].strip().split('=', 1)
            if len(partes) == 2:
                clave, valor = partes[0].strip(), partes[1].strip()
                return self.herramientas.ejecutar('memorizar', clave, valor)
        
        if 'recordar' in mensaje:
            clave = mensaje.split('recordar', 1)[1].strip()
            return self.herramientas.ejecutar('recordar', clave)
        
        return None
    
    def _responder_pregunta(self, mensaje):
        # Buscar en memoria
        similares = self.memoria.recordar(mensaje, k=2)
        if similares:
            return f"Recuerdo algo similar: {similares[0].get('salida', 'N/A')}"
        
        # Buscar en conocimiento base
        if 'capital' in mensaje:
            for key, value in self.herramientas.conocimiento_base.items():
                if key.replace('_', ' ') in mensaje:
                    return f"La {key.replace('_', ' ')} es {value}"
        
        return "Buena pregunta. No tengo suficiente información para responder."
    
    def _respuesta_generica(self, mensaje):
        # Guardar en memoria
        self.memoria.aprender([len(mensaje)], [len(mensaje) % 2])
        self.historial.append(mensaje)
        if len(self.historial) > 50:
            self.historial.pop(0)
        
        respuestas = [
            f"He recibido: {mensaje}",
            f"Entendido: {mensaje}",
            f"Procesado: {mensaje}",
            "¿Puedes repetirlo?",
            "Interesante. ¿Qué más?",
            "No estoy seguro de entender eso."
        ]
        return np.random.choice(respuestas)
    
    def chat(self):
        print(f"\n🤖 {self.nombre} - Chat Interactivo")
        print("="*40)
        print("Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas")
        print("Escribe 'salir' para terminar")
        print("="*40)
        
        while True:
            try:
                mensaje = input("\n🧑 Tú: ")
                if mensaje.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                respuesta = self.procesar(mensaje)
                print(f"🤖 {self.nombre}: {respuesta}")
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat = ChatInteractivo()
    chat.chat()
