#!/usr/bin/env python
import numpy as np
import pickle
from entrenar_xor import entrenar_xor
from entrenar_profundo import ejecutar_profundo
from entrenar_memoria import ejecutar_memoria

def menu():
    print("""
🧠 CEREBRO ZERO - MENÚ COMPLETO
================================
1. Entrenar XOR (básico) - ✅
2. Entrenar SENO (regresión polinómica) - ✅
3. Entrenar CEREBRO PROFUNDO (5 capas)
4. Cerebro con MEMORIA
5. Autoencoder (comprime datos)
6. Cerebro con VISIÓN (convolucional)
7. Aprendizaje por REFORZAMIENTO (juega)
8. Visualizar datos
9. Ver resumen de todos los cerebros
0. Salir
""")
    return input("Elige una opción: ")

def ejecutar_seno():
    print("🧠 EJECUTANDO SENO POLINÓMICO...")
    exec(open("seno_polinomico.py").read())

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
        from autoencoder import Autoencoder
        X = np.random.randn(100, 10)
        ae = Autoencoder(10, 3)
        ae.entrenar(X, epochs=500)
    elif opcion == "6":
        from convolutional import CerebroConVision
        cerebro = CerebroConVision(28)
        cerebro.resumen()
    elif opcion == "7":
        from reforzamiento import CerebroReforzado, EntornoSimple
        env = EntornoSimple()
        cerebro = CerebroReforzado(11, 2)
        cerebro.jugar(env, 100)
    elif opcion == "8":
        from visualizar import Visualizador
        from datos import GestorDatos
        X, y = GestorDatos.generar_seno(100)
        Visualizador.graficar_texto(X, y)
    elif opcion == "9":
        print("\n📊 RESUMEN DE CEREBROS:")
        print("   - red_neuronal.py: Cerebro básico (sigmoid) - ✅ XOR")
        print("   - red_profunda.py: Cerebro profundo (5 capas + ReLU)")
        print("   - memoria.py: Cerebro con memoria circular")
        print("   - seno_polinomico.py: Regresión polinómica - ✅ SENO")
        print("   - red_neuronal_relu.py: Cerebro con ReLU + salida lineal")
        print("   - autoencoder.py: Autoencoder (compresión)")
        print("   - reforzamiento.py: Q-Learning")
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
