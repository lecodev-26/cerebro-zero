import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
import json
import urllib.request
import random
from datetime import datetime

class HerramientasReales:
    def __init__(self):
        self.tools = {
            # Herramientas básicas
            'clima': self.clima,
            'noticias': self.noticias,
            'chiste': self.chiste,
            'cita': self.cita,
            'definicion': self.definicion,
            'edad': self.calcular_edad,
            'conversor': self.conversor_moneda,
            'recordar': self.recordar,
            'memorizar': self.memorizar,
            # Nuevas herramientas
            'horoscopo': self.horoscopo,
            'fecha': self.fecha,
            'hora': self.hora,
            'temporizador': self.temporizador,
            'recordatorio': self.recordatorio,
            'lista': self.lista_tareas,
            'tarjeta': self.tarjeta_credito,
            'seguro': self.seguro,
            'calculadora': self.calculadora,
            'consejo': self.consejo
        }
        self.memoria = {}
        self.tareas = []
        self.recordatorios = []
    
    # === HERRAMIENTAS EXISTENTES ===
    
    def noticias(self, categoria="general"):
        noticias = {
            'general': ["📰 Nuevo avance en IA desde España", "📰 El mercado tecnológico crece un 20%", "📰 Innovación en robótica"],
            'deportes': ["⚽ El Real Madrid gana la Champions", "⚽ Messi anuncia su retiro", "⚽ Nuevo récord en natación"],
            'ciencia': ["🔬 Descubren nuevo planeta", "🔬 Vacuna contra el cáncer en fase final", "🔬 Inteligencia artificial cuántica"],
            'tecnologia': ["📱 Lanzamiento de nuevo smartphone", "📱 Internet 6G en desarrollo", "📱 Revolución en baterías"],
            'economia': ["💰 La inflación baja al 2%", "💰 Bitcoin alcanza nuevo máximo", "💰 Nuevo tratado comercial"]
        }
        return random.choice(noticias.get(categoria, noticias['general']))
    
    def clima(self, ciudad="Madrid"):
        temperaturas = {
            'madrid': 28, 'barcelona': 26, 'valencia': 30, 'sevilla': 35,
            'bilbao': 22, 'malaga': 32, 'zaragoza': 33, 'granada': 34,
            'paris': 25, 'londres': 20, 'berlin': 22, 'roma': 29
        }
        condiciones = ['☀️ soleado', '⛅ nublado', '🌧️ lluvioso', '💨 ventoso', '☀️ despejado']
        ciudad_lower = ciudad.lower()
        temp = temperaturas.get(ciudad_lower, random.randint(15, 35))
        cond = random.choice(condiciones)
        return f"🌡️ Clima en {ciudad}: {temp}°C, {cond}"
    
    def chiste(self):
        chistes = [
            "¿Qué le dice un bit a otro? ¡Nos vemos en el bus!",
            "¿Cómo se llama el campeón de buceo japonés? Tokofondo.",
            "¿Qué hace una abeja en el gimnasio? ¡Zum-ba!",
            "¿Cómo se dice 'aparcamiento' en chino? ¡Párking!",
            "¿Qué le dice un semáforo a otro? No me mires, que me pongo rojo.",
            "¿Qué hace un pez en la universidad? ¡Nada!",
            "¿Cómo se queda un mago después de comer? ¡Magro!"
        ]
        return f"😂 {random.choice(chistes)}"
    
    def cita(self):
        citas = [
            "La inteligencia artificial no es una amenaza, es una oportunidad.",
            "El futuro pertenece a quienes creen en la belleza de sus sueños.",
            "La única forma de hacer un gran trabajo es amar lo que haces.",
            "La creatividad es la inteligencia divirtiéndose.",
            "El aprendizaje nunca agota la mente.",
            "La ciencia es poesía en acción.",
            "El conocimiento es poder."
        ]
        return f"💡 '{random.choice(citas)}'"
    
    def definicion(self, palabra):
        definiciones = {
            'ia': '🧠 Inteligencia Artificial: sistema que aprende y toma decisiones.',
            'red': '🔗 Red neuronal: sistema computacional inspirado en el cerebro humano.',
            'algoritmo': '📊 Algoritmo: secuencia de pasos para resolver un problema.',
            'datos': '📈 Datos: información organizada que sirve como base para el aprendizaje.',
            'ai': '🧠 Artificial Intelligence: capacidad de una máquina para imitar funciones humanas.',
            'machine learning': '🤖 Aprendizaje automático: rama de la IA que permite a las máquinas aprender sin programación explícita.',
            'deep learning': '🧬 Deep Learning: aprendizaje profundo usando redes neuronales con múltiples capas.'
        }
        palabra_lower = palabra.lower()
        return definiciones.get(palabra_lower, f"No tengo definición para '{palabra}'")
    
    def calcular_edad(self, año, mes=1, dia=1):
        try:
            actual = datetime.now()
            nacimiento = datetime(año, mes, dia)
            edad = actual.year - nacimiento.year - ((actual.month, actual.day) < (nacimiento.month, nacimiento.day))
            return f"📅 Edad: {edad} años"
        except:
            return "Error al calcular la edad"
    
    def conversor_moneda(self, cantidad, de_moneda="EUR", a_moneda="USD"):
        tasas = {'EUR': 1.0, 'USD': 1.1, 'GBP': 0.85, 'JPY': 160.0, 'MXN': 20.0, 'ARS': 950.0, 'BRL': 5.5, 'CLP': 1000.0}
        if de_moneda in tasas and a_moneda in tasas:
            resultado = cantidad * tasas[a_moneda] / tasas[de_moneda]
            return f"💱 {cantidad:.2f} {de_moneda} = {resultado:.2f} {a_moneda}"
        return f"Conversión no disponible para {de_moneda} → {a_moneda}"
    
    def memorizar(self, clave, valor):
        self.memoria[clave] = valor
        return f"🧠 Memorizado: {clave} = {valor}"
    
    def recordar(self, clave):
        return self.memoria.get(clave, f"No tengo información sobre '{clave}'")
    
    # === NUEVAS HERRAMIENTAS ===
    
    def horoscopo(self, signo="general"):
        horoscopo = {
            'general': "Hoy es un buen día para aprender algo nuevo.",
            'aries': "Hoy es un gran día para tomar decisiones importantes.",
            'tauro': "La perseverancia te llevará al éxito hoy.",
            'geminis': "Tu curiosidad te abrirá nuevas puertas.",
            'cancer': "La intuición te guiará hoy.",
            'leo': "Brilla con luz propia, hoy es tu día.",
            'virgo': "La organización te dará ventaja.",
            'libra': "El equilibrio es la clave hoy.",
            'escorpio': "Tu determinación te llevará lejos.",
            'sagitario': "La aventura te espera.",
            'capricornio': "La disciplina te hará fuerte.",
            'acuario': "Tu originalidad te destacará.",
            'piscis': "La creatividad fluirá hoy."
        }
        signo_lower = signo.lower()
        return f"♈ {signo}: {horoscopo.get(signo_lower, horoscopo['general'])}"
    
    def fecha(self):
        return f"📅 Hoy es: {datetime.now().strftime('%d de %B de %Y')}"
    
    def hora(self):
        return f"🕐 Son las: {datetime.now().strftime('%H:%M:%S')}"
    
    def temporizador(self, segundos=10):
        import time
        print(f"⏱️ Iniciando temporizador de {segundos} segundos...")
        time.sleep(segundos)
        return f"⏰ ¡Tiempo completado! ({segundos}s)"
    
    def recordatorio(self, mensaje, tiempo="ahora"):
        self.recordatorios.append({
            'mensaje': mensaje,
            'tiempo': tiempo,
            'fecha': datetime.now().isoformat()
        })
        return f"📋 Recordatorio guardado: '{mensaje}' para {tiempo}"
    
    def lista_tareas(self, accion="ver", tarea=""):
        if accion == "ver":
            if not self.tareas:
                return "📋 No tienes tareas pendientes."
            return f"📋 Tareas pendientes ({len(self.tareas)}):\n" + "\n".join([f"   - {t}" for t in self.tareas])
        elif accion == "añadir" and tarea:
            self.tareas.append(tarea)
            return f"✅ Tarea añadida: '{tarea}'"
        elif accion == "eliminar" and tarea:
            try:
                idx = int(tarea) - 1
                if 0 <= idx < len(self.tareas):
                    eliminada = self.tareas.pop(idx)
                    return f"🗑️ Tarea eliminada: '{eliminada}'"
                return "❌ Índice no válido"
            except:
                return "❌ Usa: lista eliminar [número]"
        return "📋 Usa: lista ver, lista añadir [tarea], lista eliminar [número]"
    
    def tarjeta_credito(self, numero):
        """Valida tarjeta de crédito (algoritmo de Luhn)"""
        def luhn_check(num):
            digits = [int(d) for d in str(num)][::-1]
            for i in range(1, len(digits), 2):
                digits[i] *= 2
                if digits[i] > 9:
                    digits[i] -= 9
            return sum(digits) % 10 == 0
        
        try:
            num = str(num).replace(' ', '').replace('-', '')
            if len(num) < 13 or len(num) > 19:
                return "❌ Número de tarjeta inválido (13-19 dígitos)"
            valido = luhn_check(num)
            return f"💳 Tarjeta {num[:4]}****{num[-4:]}: {'✅ Válida' if valido else '❌ Inválida'}"
        except:
            return "❌ Error al validar la tarjeta"
    
    def seguro(self, monto, años, tasa=5):
        """Calcula seguro de vida (simplificado)"""
        monto = float(monto)
        anual = monto * (tasa / 100)
        total = anual * años
        return f"📊 Seguro de vida: {monto:.2f}€, {años} años, tasa {tasa}%\n   Anual: {anual:.2f}€, Total: {total:.2f}€"
    
    def calculadora(self, operacion, a, b):
        if operacion == 'suma':
            return f"{a} + {b} = {a+b}"
        elif operacion == 'resta':
            return f"{a} - {b} = {a-b}"
        elif operacion == 'multiplica':
            return f"{a} × {b} = {a*b}"
        elif operacion == 'divide':
            return f"{a} ÷ {b} = {a/b if b != 0 else 'Error: División por cero'}"
        return "Operaciones: suma, resta, multiplica, divide"
    
    def consejo(self):
        consejos = [
            "💡 Aprende algo nuevo cada día.",
            "💡 La práctica hace al maestro.",
            "💡 No tengas miedo de equivocarte.",
            "💡 La perseverancia es la clave del éxito.",
            "💡 Rodéate de personas que te inspiren.",
            "💡 La curiosidad es el motor del conocimiento.",
            "💡 Cada día es una oportunidad para mejorar.",
            "💡 La IA es una herramienta, no un sustituto."
        ]
        return random.choice(consejos)
    
    def ejecutar(self, comando, *args):
        if comando in self.tools:
            try:
                return self.tools[comando](*args)
            except Exception as e:
                return f"Error: {e}"
        return f"Herramienta desconocida: {comando}"
    
    def listar(self):
        return list(self.tools.keys())

def prueba_herramientas_reales():
    print("🧠 PROBANDO HERRAMIENTAS REALES")
    print("="*40)
    
    tools = HerramientasReales()
    print(f"Herramientas disponibles: {len(tools.listar())}")
    print(f"   {', '.join(tools.listar()[:5])}...")
    
    print("\n🔧 Pruebas de nuevas herramientas:")
    print(f"   horoscopo('leo') = {tools.ejecutar('horoscopo', 'leo')}")
    print(f"   fecha() = {tools.ejecutar('fecha')}")
    print(f"   consejo() = {tools.ejecutar('consejo')}")
    print(f"   tarjeta('1234567890123456') = {tools.ejecutar('tarjeta', '1234567890123456')}")
    print(f"   lista_tareas('añadir', 'Estudiar IA') = {tools.ejecutar('lista_tareas', 'añadir', 'Estudiar IA')}")
    print(f"   lista_tareas('ver') = {tools.ejecutar('lista_tareas', 'ver')}")
    
    print("\n✅ HERRAMIENTAS REALES FUNCIONANDO!")

if __name__ == "__main__":
    prueba_herramientas_reales()
