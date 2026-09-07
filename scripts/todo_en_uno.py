#!/usr/bin/env python
import sys
sys.path.append('..')

import numpy as np
from entrenadores.entrenar_xor import entrenar_xor
from entrenadores.entrenar_premium import ejecutar_premium
from entrenadores.entrenar_profundo import ejecutar_profundo
from entrenadores.entrenar_memoria import ejecutar_memoria
from utils.autoencoder import ejecutar_autoencoder
from entrenadores.entrenar_gigante import ejecutar_gigante

def menu():
    print("""
🧠 CEREBRO ZERO - MENÚ COMPLETO
================================
1. XOR (básico) - ✅
2. SENO - ✅
3. Cargar cerebro - ✅
4. Memoria - ✅
5. CEREBRO PREMIUM - ✅
6. Autoencoder - ✅
7. Visión - ✅
8. Reforzamiento - ✅
9. 🆕 CEREBRO GIGANTE (256+256+256) - 🔥
0. Salir
""")
    return input("Elige una opción: ")

def ejecutar_seno():
    exec(open("../datos/seno_polinomico.py").read())

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
        ejecutar_autoencoder()
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
        ejecutar_gigante()
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
