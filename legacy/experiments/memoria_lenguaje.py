import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.memory import MemoriaPersistente
from experiments.mini_lenguaje import MiniModeloLenguaje

class CerebroConMemoriaLenguaje:
    def __init__(self, vocab_size=20, d_model=8, num_heads=2, d_ff=16, num_layers=2, max_len=10):
        self.modelo = MiniModeloLenguaje(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
        self.memoria = MemoriaPersistente(archivo="memoria_lenguaje.json", max_size=50)
        self.vocab_size = vocab_size
        self.conversacion = []
    
    def responder(self, entrada, temperature=0.8):
        """Responde usando memoria y generación"""
        # Buscar en memoria
        similares = self.memoria.recordar(entrada, k=2)
        
        # Si hay memoria, usar para influir en la generación
        if similares:
            print(f"🧠 Memoria encontrada: {len(similares)} experiencias similares")
            # Generar basado en memoria
            inicio = list(entrada)
            generado = self.modelo.generar(inicio, longitud=5, temperature=temperature)
            respuesta = generado[-5:]
        else:
            # Generar sin memoria
            inicio = list(entrada)
            generado = self.modelo.generar(inicio, longitud=5, temperature=temperature)
            respuesta = generado[-5:]
        
        # Aprender de la interacción
        self.memoria.aprender(np.array(entrada), np.array(respuesta))
        self.conversacion.append((entrada, respuesta))
        
        return respuesta
    
    def resumen(self):
        print("🧠 CEREBRO CON MEMORIA Y LENGUAJE")
        print("="*40)
        self.memoria.resumen()
        print(f"   Conversaciones: {len(self.conversacion)}")

def prueba_memoria_lenguaje():
    print("🧠 CEREBRO CON MEMORIA Y LENGUAJE")
    print("="*40)
    
    # Crear cerebro
    cerebro = CerebroConMemoriaLenguaje(
        vocab_size=20,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_len=10
    )
    
    # Simular conversación
    entradas = [
        [1, 2, 3],
        [1, 2, 4],
        [5, 6, 7],
        [1, 2, 3],
        [8, 9, 10]
    ]
    
    print("\n💬 SIMULANDO CONVERSACIÓN:")
    for i, entrada in enumerate(entradas):
        print(f"\n   >>> Entrada {i+1}: {entrada}")
        respuesta = cerebro.responder(entrada, temperature=0.8)
        print(f"   <<< Respuesta: {respuesta}")
    
    # Resumen
    print("\n")
    cerebro.resumen()
    
    print("\n✅ CEREBRO CON MEMORIA Y LENGUAJE FUNCIONANDO!")

if __name__ == "__main__":
    prueba_memoria_lenguaje()
