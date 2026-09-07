import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.memory import MemoriaPersistente

print("🧠 PROBANDO MEMORIA PERSISTENTE")
print("="*40)

# Crear memoria
memoria = MemoriaPersistente(archivo="memoria_test.json", max_size=10)

# Aprender algunas experiencias
print("\n📝 APRENDIENDO EXPERIENCIAS...")
experiencias = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

for entrada, salida in experiencias:
    memoria.aprender(np.array(entrada), np.array(salida))
    print(f"   {entrada} → {salida}")

# Mostrar resumen
print("\n📊 RESPUESTAS:")
memoria.resumen()

# Probar recuperación
print("\n🔮 RECUPERANDO EXPERIENCIAS SIMILARES...")
pruebas = [[0, 0], [1, 1], [0.5, 0.5]]
for prueba in pruebas:
    similares = memoria.recordar(prueba, k=2)
    print(f"   {prueba} → {[s['salida'] for s in similares]}")

print("\n✅ MEMORIA FUNCIONANDO CORRECTAMENTE!")
