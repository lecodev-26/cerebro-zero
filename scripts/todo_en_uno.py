#!/usr/bin/env python
import sys
sys.path.append('..')

import numpy as np
from entrenadores.entrenar_xor import entrenar_xor
from entrenadores.entrenar_premium import ejecutar_premium
from entrenadores.entrenar_profundo import ejecutar_profundo
from entrenadores.entrenar_memoria import ejecutar_memoria

def menu():
    print("""
🧠 CEREBRO ZERO - MENÚ COMPLETO
================================
1. Entrenar XOR (básico) - ✅
2. Entrenar SENO (regresión polinómica) - ✅
3. Cargar cerebro guardado - ✅
4. Cerebro con MEMORIA - ✅
5. CEREBRO PREMIUM (Softmax + Adam + Dropout + BatchNorm) - 🆕
6. Autoencoder - ⏳
7. Visión - ⏳
8. Reforzamiento - ⏳
9. Resumen
0. Salir
""")
    return input("Elige una opción: ")

def ejecutar_seno():
    exec(open("datos/seno_polinomico.py").read())

def ejecutar(opcion):
    if opcion == "1":
        entrenar_xor()
    elif opcion == "2":
        ejecutar_seno()
    elif opcion == "3":
        ejecutar_profundo()
    elif opcion == "4":
        ejecutar_memoria()
    elif opcion == "5":
        ejecutar_premium()
    elif opcion == "6":
        from utils.autoencoder import Autoencoder
        X = np.random.randn(100, 10)
        ae = Autoencoder(10, 3)
        ae.entrenar(X, epochs=500)
    elif opcion == "7":
        from utils.convolutional import CerebroConVision
        cerebro = CerebroConVision(28)
        cerebro.resumen()
    elif opcion == "8":
        from utils.reforzamiento import CerebroReforzado, EntornoSimple
        env = EntornoSimple()
        cerebro = CerebroReforzado(11, 2)
        cerebro.jugar(env, 100)
    elif opcion == "9":
        print("\n📊 RESUMEN DE CEREBROS:")
        print("   - cerebros/red_neuronal.py: Cerebro básico (sigmoid)")
        print("   - cerebros/red_neuronal_premium.py: Cerebro premium")
        print("   - cerebros/red_profunda.py: Cerebro profundo")
        print("   - cerebros/memoria.py: Cerebro con memoria")
        print("   - datos/seno_polinomico.py: Regresión polinómica")
        print("   - utils/autoencoder.py: Autoencoder")
        print("   - utils/convolutional.py: Visión")
        print("   - utils/reforzamiento.py: Q-Learning")
    elif opcion == "0":
        print("👋 Hasta luego, Manuel!")
        return False
    return True

if __name__ == "__main__":
    print("\n🧠 CEREBRO ZERO - SISTEMA COMPLETO")
    print("="*40)
    while True:
        opcion = menu()
        if not ejecutar(opcion):
            break
        input("\nPresiona ENTER para continuar...")
