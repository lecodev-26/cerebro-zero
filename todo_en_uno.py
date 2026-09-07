#!/usr/bin/env python
import sys
sys.path.append('..')
import numpy as np
import os

# Importar todos los entrenadores
from entrenadores.entrenar_xor import entrenar_xor
from entrenadores.entrenar_premium import ejecutar_premium
from entrenadores.entrenar_profundo import ejecutar_profundo
from entrenadores.entrenar_memoria import ejecutar_memoria
from utils.autoencoder import ejecutar_autoencoder
from entrenadores.entrenar_gigante import ejecutar_gigante
from entrenadores.entrenar_maximo import ejecutar_maximo
from entrenadores.entrenar_dios import ejecutar_dios
from entrenadores.entrenar_mnist import ejecutar_mnist

def menu():
    print("""
🧠 CEREBRO ZERO - SISTEMA COMPLETO
====================================
🧬 CEREBROS BÁSICOS
   1. XOR (básico) - ✅
   2. SENO (regresión polinómica) - ✅
   3. Cargar cerebro guardado - ✅
   4. Memoria - ✅

🚀 CEREBROS AVANZADOS
   5. PREMIUM (Softmax + Adam + Dropout + BatchNorm) - ✅
   6. Autoencoder (PCA) - ✅
   7. Visión - ✅
   8. Reforzamiento (Q-Learning) - ✅

🔥 CEREBROS GIGANTES
   9. GIGANTE (64+64) - ✅
   10. MÁXIMO (512) - ✅
   11. DIOS (1000) - ✅
   12. MNIST (1024 neuronas - 79% precisión) - 🆕

📊 OPCIONES
   13. Resumen completo
   0. Salir
""")
    return input("Elige una opción: ")

def ejecutar_seno():
    exec(open("../datos/seno_polinomico.py").read())

def ejecutar_resumen():
    print("\n📊 RESUMEN COMPLETO DEL PROYECTO")
    print("="*50)
    print("🧠 CEREBROS CREADOS:")
    print("   1. XOR (8 neuronas) - Error: 0.0001")
    print("   2. SENO (regresión polinómica) - Funciona")
    print("   3. Memoria (secuencias) - Funciona")
    print("   4. PREMIUM (Softmax + Adam + Dropout + BatchNorm) - Funciona")
    print("   5. Autoencoder (PCA) - Funciona")
    print("   6. Visión (convolucional) - Funciona")
    print("   7. Reforzamiento (Q-Learning) - Funciona")
    print("   8. GIGANTE (64+64) - Funciona")
    print("   9. MÁXIMO (512) - Funciona")
    print("   10. DIOS (1000) - Funciona")
    print("   11. MNIST (1024 neuronas) - 78.95% precisión 🆕")
    print("="*50)
    print("📁 MODELOS GUARDADOS:")
    modelos = os.listdir("../modelos_guardados/")
    for m in modelos:
        if m.endswith(".pkl"):
            size = os.path.getsize(f"../modelos_guardados/{m}")
            print(f"   ✅ {m} ({size:,} bytes)")
    print("="*50)
    print("🔥 TOTAL DE PARÁMETROS ENTRENADOS:")
    print("   - MNIST: 814,090 parámetros")
    print("   - DIOS: 12,001 parámetros")
    print("   - MÁXIMO: 5,632 parámetros")
    print("   - Total: ~831,723 parámetros")
    print("="*50)

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
    elif opcion == "10":
        ejecutar_maximo()
    elif opcion == "11":
        ejecutar_dios()
    elif opcion == "12":
        ejecutar_mnist()
    elif opcion == "13":
        ejecutar_resumen()
    elif opcion == "0":
        print("👋 Hasta luego, Manuel! 🧠🔥")
        return False
    else:
        print("❌ Opción no válida")
        return True
    return True

if __name__ == "__main__":
    print("\n🔥 CEREBRO ZERO - SISTEMA COMPLETO")
    print("="*40)
    print("   ¡Bienvenido, Manuel! Has creado:")
    print("   - 12 cerebros diferentes")
    print("   - 1 cerebro de 1024 neuronas (79% precisión)")
    print("   - Más de 800,000 parámetros entrenados")
    print("="*40)
    while True:
        opcion = menu()
        if not ejecutar(opcion):
            break
        input("\nPresiona ENTER para continuar...")
