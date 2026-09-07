import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
import subprocess

class ReconocimientoVoz:
    def __init__(self):
        self.disponible = False
        self._verificar_dependencias()
    
    def _verificar_dependencias(self):
        """Verifica si las herramientas de voz están disponibles"""
        try:
            # Verificar si termux-API está instalado
            result = subprocess.run(['which', 'termux-microphone-record'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.disponible = True
                print("🎤 Reconocimiento de voz disponible (Termux-API)")
            else:
                print("⚠️ Termux-API no encontrado. Instala: pkg install termux-api")
                print("   O usa el modo texto para probar el reconocimiento simulado")
        except:
            print("⚠️ No se pudo verificar el reconocimiento de voz")
    
    def grabar(self, duracion=3):
        """Graba audio desde el micrófono"""
        if not self.disponible:
            return self._simular_grabacion()
        
        try:
            archivo = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            archivo.close()
            
            print(f"🎤 Grabando {duracion} segundos...")
            subprocess.run([
                'termux-microphone-record', 
                '-f', archivo.name,
                '-d', str(duracion),
                '-l', '0'
            ], check=True)
            
            print("✅ Grabación completada")
            return archivo.name
        except Exception as e:
            print(f"❌ Error grabando: {e}")
            return None
    
    def transcribir(self, archivo_audio):
        """Transcribe audio a texto (simulado sin API externa)"""
        if not archivo_audio:
            return None
        
        # Simulación de transcripción (no tenemos API de voz en Termux)
        respuestas = [
            "hola",
            "cómo estás",
            "qué hora es",
            "clima en madrid",
            "ayuda",
            "adiós",
            "dado 6",
            "moneda",
            "traducir hola ingles",
            "memorizar color azul"
        ]
        
        import random
        return random.choice(respuestas)
    
    def _simular_grabacion(self):
        """Simula una grabación para pruebas"""
        print("🎤 SIMULANDO GRABACIÓN (sin micrófono)")
        return "hola mundo"
    
    def escuchar(self, duracion=3):
        """Escucha y transcribe voz (flujo completo)"""
        print("🎤 Escuchando...")
        
        # Simular escucha
        import time
        for i in range(duracion):
            print(f"   .", end='', flush=True)
            time.sleep(1)
        print()
        
        # Transcribir
        if self.disponible:
            archivo = self.grabar(duracion)
            texto = self.transcribir(archivo)
        else:
            texto = self._simular_grabacion()
        
        print(f"📝 Texto: {texto}")
        return texto

def prueba_voz():
    print("🧠 PROBANDO RECONOCIMIENTO DE VOZ")
    print("="*30)
    
    voz = ReconocimientoVoz()
    texto = voz.escuchar(duracion=2)
    print(f"✅ Transcripción: {texto}")

if __name__ == "__main__":
    prueba_voz()
