import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
from memory.memory import MemoriaPersistente
from tools.herramientas_avanzadas import HerramientasAvanzadas
from tools.herramientas_extra import HerramientasExtra

class ChatMejorado:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="chat_mejorado.json", max_size=100)
        self.herramientas = HerramientasAvanzadas()
        self.herramientas_extra = HerramientasExtra()
        self.nombre = "Cerebro Zero Pro"
        self.historial = []
        self.usuario = "usuario"
    
    def procesar(self, mensaje):
        mensaje = mensaje.strip()
        
        # Comandos especiales
        if mensaje.startswith('/'):
            return self._procesar_comando(mensaje)
        
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
        
        # Respuesta por defecto
        return self._respuesta_inteligente(mensaje)
    
    def _detectar_herramienta_avanzada(self, mensaje):
        mensaje_lower = mensaje.lower()
        herramientas = self.herramientas.listar()
        
        for tool in herramientas:
            if tool in mensaje_lower:
                # Extraer parámetros
                numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
                if tool in ['sumar', 'restar', 'multiplicar', 'dividir', 'potencia']:
                    if len(numeros) >= 2:
                        return f"{self.herramientas.ejecutar(tool, float(numeros[0]), float(numeros[1]))}"
                elif tool == 'raiz':
                    if numeros:
                        return f"√({numeros[0]}) = {self.herramientas.ejecutar('raiz', float(numeros[0])):.4f}"
                elif tool == 'traducir':
                    palabras = mensaje.replace('traducir', '').strip().split()
                    if len(palabras) >= 2:
                        texto = palabras[0]
                        idioma = palabras[1] if len(palabras) > 1 else 'ingles'
                        return self.herramientas.ejecutar('traducir', texto, idioma)
                elif tool == 'memorizar':
                    partes = mensaje.split('memorizar', 1)[1].strip().split('=', 1)
                    if len(partes) == 2:
                        clave, valor = partes[0].strip(), partes[1].strip()
                        return self.herramientas.ejecutar('memorizar', clave, valor)
                elif tool == 'recordar':
                    clave = mensaje.split('recordar', 1)[1].strip()
                    return self.herramientas.ejecutar('recordar', clave)
                elif tool in ['hora', 'fecha']:
                    return self.herramientas.ejecutar(tool)
                elif tool == 'buscar':
                    query = mensaje.replace('buscar', '').strip()
                    return self.herramientas.ejecutar('buscar', query)
        
        return None
    
    def _detectar_herramienta_extra(self, mensaje):
        mensaje_lower = mensaje.lower()
        herramientas = self.herramientas_extra.listar()
        
        for tool in herramientas:
            if tool in mensaje_lower:
                numeros = re.findall(r'[-+]?\d*\.?\d+', mensaje)
                if tool == 'clima':
                    ciudad = mensaje.replace('clima', '').strip() or 'Madrid'
                    return self.herramientas_extra.ejecutar('clima', ciudad)
                elif tool == 'web':
                    url = mensaje.replace('web', '').strip() or 'ejemplo.com'
                    return self.herramientas_extra.ejecutar('web', url)
                elif tool == 'calcular_edad':
                    if numeros:
                        return self.herramientas_extra.ejecutar('calcular_edad', int(numeros[0]))
                elif tool == 'conversor_moneda':
                    palabras = mensaje.replace('conversor_moneda', '').strip().split()
                    if len(palabras) >= 3:
                        return self.herramientas_extra.ejecutar('conversor_moneda', 
                            float(palabras[0]), palabras[1], palabras[2])
                elif tool == 'dado':
                    caras = int(numeros[0]) if numeros else 6
                    return f"🎲 Dado: {self.herramientas_extra.ejecutar('dado', caras)}"
                elif tool == 'moneda':
                    return f"🪙 Moneda: {self.herramientas_extra.ejecutar('moneda')}"
                elif tool == 'numero_aleatorio':
                    min_val = int(numeros[0]) if len(numeros) > 0 else 1
                    max_val = int(numeros[1]) if len(numeros) > 1 else 100
                    return f"🔢 Número: {self.herramientas_extra.ejecutar('numero_aleatorio', min_val, max_val)}"
        
        return None
    
    def _procesar_comando(self, mensaje):
        partes = mensaje.split()
        comando = partes[0][1:]
        
        comandos = {
            'ayuda': f"Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /herramientas_extra",
            'historial': f"Últimos mensajes: {self.historial[-5:]}",
            'olvidar': self._olvidar_memoria(),
            'nombre': f"Me llamo {self.nombre}",
            'herramientas': f"Herramientas avanzadas: {self.herramientas.listar()}",
            'herramientas_extra': f"Herramientas extra: {self.herramientas_extra.listar()}"
        }
        return comandos.get(comando, f"Comando desconocido: {comando}")
    
    def _olvidar_memoria(self):
        self.memoria.olvidar()
        return "🧠 Memoria borrada"
    
    def _respuesta_inteligente(self, mensaje):
        respuestas = [
            f"He recibido: {mensaje}",
            f"Entendido: {mensaje}",
            "¿Puedes ser más específico?",
            "Interesante. ¿Qué más?",
            "No estoy seguro de entender eso.",
            "¿Podrías reformularlo?",
            "Entiendo. Continuemos.",
            "Dime más sobre eso."
        ]
        return np.random.choice(respuestas)
    
    def chat(self):
        print(f"\n🤖 {self.nombre} - Chat Mejorado")
        print("="*40)
        print("Comandos: /ayuda, /historial, /olvidar, /nombre, /herramientas, /herramientas_extra")
        print("Escribe 'salir' para terminar")
        print("="*40)
        print("\n💡 EJEMPLOS DE USO:")
        print("   - 'clima Madrid' → Clima")
        print("   - 'dado 6' → Lanzar dado")
        print("   - 'traducir hola ingles' → Traducción")
        print("   - 'memorizar mi_color = azul' → Memorizar")
        print("   - 'recordar mi_color' → Recordar")
        print("   - 'conversor_moneda 100 EUR USD' → Conversión")
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
    chat = ChatMejorado()
    chat.chat()
