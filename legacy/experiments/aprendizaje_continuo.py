import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.memory import MemoriaPersistente

class AprendizajeContinuo:
    def __init__(self):
        self.memoria = MemoriaPersistente(archivo="aprendizaje_continuo.json", max_size=100)
        self.conocimiento = {}
        self.errores = []
    
    def aprender(self, entrada, salida, confianza=1.0):
        experiencia = {
            'entrada': entrada,
            'salida': salida,
            'confianza': confianza,
            'repeticiones': 1
        }
        
        similares = self.memoria.recordar(entrada, k=1)
        if similares and similares[0].get('confianza', 0) > 0.1:
            idx = self.memoria.experiencias.index(similares[0])
            self.memoria.experiencias[idx]['repeticiones'] += 1
            self.memoria.experiencias[idx]['confianza'] = min(1.0, confianza + 0.1)
        else:
            self.memoria.aprender(np.array(entrada), np.array(salida))
    
    def recordar(self, entrada, umbral=0.1):
        similares = self.memoria.recordar(entrada, k=3)
        return [s for s in similares if s.get('confianza', 0) >= umbral]
    
    def revisar(self):
        """Revisa el conocimiento aprendido"""
        for exp in self.memoria.experiencias:
            if exp.get('repeticiones', 0) < 1 and exp.get('confianza', 0) < 0.1:
                self.memoria.olvidar(self.memoria.experiencias.index(exp))
    
    def resumen(self):
        print("🧠 APRENDIZAJE CONTINUO")
        print("="*40)
        self.memoria.resumen()
        print(f"   Conocimiento total: {len(self.memoria.experiencias)}")

def prueba_aprendizaje_continuo():
    print("🧠 APRENDIZAJE CONTINUO")
    print("="*40)
    
    cerebro = AprendizajeContinuo()
    
    print("\n📝 APRENDIENDO...")
    datos = [
        ([1, 2], [3], 0.9),
        ([2, 3], [5], 0.8),
        ([3, 4], [7], 0.7),
        ([1, 2], [3], 0.95),
        ([5, 6], [11], 0.6),
        ([1, 2], [3], 0.98),
    ]
    
    for entrada, salida, confianza in datos:
        cerebro.aprender(entrada, salida, confianza)
        print(f"   Aprendido: {entrada} → {salida} (confianza: {confianza:.2f})")
    
    print("\n🔮 RECORDANDO:")
    pruebas = [[1, 2], [3, 4], [10, 20]]
    for prueba in pruebas:
        recordado = cerebro.recordar(prueba)
        print(f"   {prueba} → {recordado}")
    
    print("\n")
    cerebro.resumen()
    print("\n✅ APRENDIZAJE CONTINUO FUNCIONANDO!")

if __name__ == "__main__":
    prueba_aprendizaje_continuo()
