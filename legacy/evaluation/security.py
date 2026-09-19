"""
Sistema de seguridad para Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from datetime import datetime

class SecuritySystem:
    def __init__(self):
        # Palabras prohibidas
        self.palabras_prohibidas = [
            'rm -rf', 'format', 'delete', 'drop table',
            'sudo', 'chmod 777', 'eval(', 'exec(',
            'os.system', 'subprocess', '__import__'
        ]
        
        # Patrones peligrosos
        self.patrones_peligrosos = [
            r'rm\s+-rf',
            r'format\s+[a-z]:',
            r'del\s+/[fqs]',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'os\.system',
            r'subprocess\.',
        ]
        
        # Comandos permitidos (whitelist)
        self.comandos_permitidos = [
            'calculadora', 'fecha', 'hora', 'contar',
            'mayusculas', 'minusculas', 'longitud',
            'invertir', 'sumar_lista', 'promedio'
        ]
        
        # Límites
        self.max_longitud = 1000
        self.max_repeticiones = 10
        
        # Historial de intentos
        self.intentos_bloqueados = []
        self.estadisticas = {
            'total_verificaciones': 0,
            'bloqueados': 0,
            'permitidos': 0
        }
    
    def verificar(self, entrada):
        """
        Verifica si una entrada es segura
        """
        self.estadisticas['total_verificaciones'] += 1
        
        # 1. Verificar longitud
        if len(entrada) > self.max_longitud:
            self._bloquear(entrada, "Entrada demasiado larga")
            return False, "Entrada demasiado larga"
        
        # 2. Verificar palabras prohibidas
        entrada_lower = entrada.lower()
        for palabra in self.palabras_prohibidas:
            if palabra.lower() in entrada_lower:
                self._bloquear(entrada, f"Palabra prohibida: {palabra}")
                return False, f"Palabra prohibida detectada: {palabra}"
        
        # 3. Verificar patrones peligrosos
        for patron in self.patrones_peligrosos:
            if re.search(patron, entrada_lower):
                self._bloquear(entrada, f"Patrón peligroso: {patron}")
                return False, f"Patrón peligroso detectado"
        
        # 4. Verificar caracteres extraños
        if self._tiene_caracteres_sospechosos(entrada):
            self._bloquear(entrada, "Caracteres sospechosos")
            return False, "Caracteres sospechosos detectados"
        
        # 5. Todo OK
        self.estadisticas['permitidos'] += 1
        return True, "Entrada segura"
    
    def _tiene_caracteres_sospechosos(self, entrada):
        """
        Detecta caracteres sospechosos
        """
        # Permitir: letras, números, espacios, puntuación básica
        patron_permitido = r'^[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s\.,;:!?¡¿()\[\]{}+\-*/=%$€@#&\'"<>|_~^°]+$'
        return not re.match(patron_permitido, entrada)
    
    def _bloquear(self, entrada, razon):
        """
        Registra un intento bloqueado
        """
        self.estadisticas['bloqueados'] += 1
        self.intentos_bloqueados.append({
            'entrada': entrada[:100],  # Truncar
            'razon': razon,
            'fecha': datetime.now().isoformat()
        })
    
    def verificar_herramienta(self, herramienta):
        """
        Verifica si una herramienta está permitida
        """
        return herramienta in self.comandos_permitidos
    
    def obtener_intentos_bloqueados(self, limite=5):
        """Obtiene los últimos intentos bloqueados"""
        return self.intentos_bloqueados[-limite:]
    
    def estadisticas_seguridad(self):
        """Muestra estadísticas de seguridad"""
        print("\n🔒 ESTADÍSTICAS DE SEGURIDAD")
        print("="*40)
        print(f"   Total verificaciones: {self.estadisticas['total_verificaciones']}")
        print(f"   Permitidos: {self.estadisticas['permitidos']}")
        print(f"   Bloqueados: {self.estadisticas['bloqueados']}")
        
        if self.intentos_bloqueados:
            print(f"\n   Últimos bloqueos:")
            for intento in self.intentos_bloqueados[-3:]:
                print(f"      - {intento['razon']}: {intento['entrada'][:50]}...")
    
    def __repr__(self):
        return f"SecuritySystem(bloqueados={self.estadisticas['bloqueados']})"

def prueba_seguridad():
    print("🧠 PROBANDO SISTEMA DE SEGURIDAD")
    print("="*40)
    
    seguridad = SecuritySystem()
    
    # Casos de prueba
    casos = [
        ("hola", True),
        ("¿Qué hora es?", True),
        ("¿Cuánto es 5 + 3?", True),
        ("rm -rf /", False),
        ("eval('print(1)')", False),
        ("os.system('ls')", False),
        ("a" * 2000, False),
        ("borra todos los archivos", True),  # No es comando técnico
        ("__import__('os')", False),
    ]
    
    print("\n📝 Verificando entradas:")
    for entrada, esperado in casos:
        seguro, mensaje = seguridad.verificar(entrada)
        estado = "✅" if seguro == esperado else "❌"
        print(f"\n   {estado} Entrada: {entrada[:50]}")
        print(f"      Seguro: {seguro}")
        print(f"      Mensaje: {mensaje}")
    
    seguridad.estadisticas_seguridad()
    
    print("\n✅ SISTEMA DE SEGURIDAD FUNCIONANDO!")

if __name__ == "__main__":
    prueba_seguridad()
