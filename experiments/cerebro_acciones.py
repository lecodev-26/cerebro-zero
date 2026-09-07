import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import re
from memory.memory import MemoriaPersistente
from tools.herramientas import Herramientas

class CerebroConAcciones:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="memoria_acciones.json", max_size=50)
        self.herramientas = Herramientas()
        self.conversacion = []
        
        self.intenciones = {
            'sumar': r'suma|sumar|\+',
            'restar': r'resta|restar|-',
            'multiplicar': r'multiplica|multiplicar|\*',
            'dividir': r'divide|dividir|/',
            'potencia': r'potencia|elevar|\^',
            'raiz': r'raiz|sqrt|√',
            'contar': r'contar|cuantos'
        }
    
    def detectar_intencion(self, entrada):
        entrada_str = str(entrada).lower()
        for intencion, patron in self.intenciones.items():
            if re.search(patron, entrada_str):
                return intencion
        return None
    
    def extraer_numeros(self, entrada):
        numeros = re.findall(r'[-+]?\d*\.?\d+', str(entrada))
        return [float(n) for n in numeros] if numeros else []
    
    def procesar(self, entrada):
        entrada_str = str(entrada)
        intencion = self.detectar_intencion(entrada_str)
        numeros = self.extraer_numeros(entrada_str)
        
        if intencion and numeros:
            if intencion in ['sumar', 'restar', 'multiplicar', 'dividir', 'potencia']:
                if len(numeros) >= 2:
                    resultado = self.herramientas.ejecutar(intencion, numeros[0], numeros[1])
                    respuesta = f"{intencion}({numeros[0]}, {numeros[1]}) = {resultado}"
                    # Guardar en memoria
                    self.memoria.aprender(numeros, [resultado])
                    return respuesta
            
            elif intencion == 'raiz':
                resultado = self.herramientas.ejecutar('raiz', numeros[0], numeros[1] if len(numeros) > 1 else 2)
                respuesta = f"raiz({numeros[0]}) = {resultado}"
                self.memoria.aprender(numeros, [resultado])
                return respuesta
            
            elif intencion == 'contar':
                resultado = len(numeros)
                respuesta = f"Hay {resultado} números"
                self.memoria.aprender(numeros, [resultado])
                return respuesta
        
        return f"Procesado: {entrada_str}"
    
    def resumen(self):
        print("🧠 CEREBRO CON ACCIONES")
        print("="*40)
        self.memoria.resumen()
        print(f"   Herramientas: {self.herramientas.listar()}")
        print(f"   Conversaciones: {len(self.conversacion)}")

def prueba_cerebro_acciones():
    print("🧠 CEREBRO CON ACCIONES")
    print("="*40)
    
    cerebro = CerebroConAcciones()
    
    pruebas = [
        "suma 5 y 3",
        "multiplica 4 por 7",
        "potencia 2 elevado a 8",
        "raiz cuadrada de 16",
        "contar cuantos numeros hay en 1 2 3 4 5",
        "suma 10 y 20",
        "resta 100 menos 25",
        "hola mundo"
    ]
    
    print("\n💬 PROCESANDO ENTRADAS:")
    for i, prueba in enumerate(pruebas):
        print(f"\n   >>> Entrada {i+1}: {prueba}")
        resultado = cerebro.procesar(prueba)
        print(f"   <<< Respuesta: {resultado}")
    
    print("\n")
    cerebro.resumen()
    print("\n✅ CEREBRO CON ACCIONES FUNCIONANDO!")

if __name__ == "__main__":
    prueba_cerebro_acciones()
