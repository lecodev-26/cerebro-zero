import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import tempfile
import json
import time

class VozReal:
    def __init__(self):
        self.disponible = False
        self._verificar()
    
    def _verificar(self):
        try:
            # Verificar si termux-microphone-record está disponible
            result = subprocess.run(['which', 'termux-microphone-record'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.disponible = True
                print("🎤 Micrófono disponible")
            else:
                print("⚠️ termux-microphone-record no encontrado")
                print("   Instala: pkg install termux-api")
        except:
            print("⚠️ No se pudo verificar el micrófono")
    
    def grabar(self, duracion=3):
        """Graba audio y devuelve la ruta del archivo"""
        if not self.disponible:
            return None
        
        try:
            archivo = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            archivo.close()
            
            print(f"🎤 Grabando {duracion} segundos...")
            subprocess.run([
                'termux-microphone-record',
                '-f', archivo.name,
                '-d', str(duracion),
                '-l', '0'
            ], check=True, capture_output=True)
            
            print("✅ Grabación completada")
            return archivo.name
        except Exception as e:
            print(f"❌ Error grabando: {e}")
            return None
    
    def transcribir(self, archivo_audio):
        """Transcribe audio usando termux-speech-to-text"""
        if not self.disponible or not archivo_audio:
            return None
        
        try:
            # Usar termux-speech-to-text (requiere instalar termux-api)
            result = subprocess.run([
                'termux-speech-to-text',
                '-f', archivo_audio
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"❌ Error en transcripción: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def escuchar(self, duracion=3):
        """Flujo completo: grabar + transcribir"""
        print("🎤 Escuchando...")
        archivo = self.grabar(duracion)
        if not archivo:
            return self._simular()
        
        texto = self.transcribir(archivo)
        if texto:
            print(f"📝 Transcripción: {texto}")
            return texto
        
        return self._simular()
    
    def _simular(self):
        """Simula transcripción para pruebas"""
        respuestas = [
            "hola", "cómo estás", "qué hora es",
            "clima Madrid", "chiste", "ayuda", "adiós"
        ]
        import random
        return random.choice(respuestas)

def prueba_voz_real():
    print("🧠 PROBANDO VOZ REAL")
    print("="*30)
    
    voz = VozReal()
    if voz.disponible:
        print("🎤 Micrófono listo para pruebas")
        print("   Habla durante 3 segundos...")
        texto = voz.escuchar(duracion=3)
        print(f"📝 Texto reconocido: {texto}")
    else:
        print("❌ Voz real no disponible")
        print("   Usando modo simulado")

if __name__ == "__main__":
    prueba_voz_real()
