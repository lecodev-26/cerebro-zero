#!/usr/bin/env python
import numpy as np

def menu():
    print("""
🧠 CEREBRO ZERO - MENÚ COMPLETO
================================
1. Entrenar XOR (básico)
2. Entrenar SENO (datos reales)
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

def ejecutar(opcion):
    if opcion == "1":
        from entrenar import Cerebro, Entrenador
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[1],[1],[0]])
        cerebro = Cerebro([2, 4, 1])
        entrenador = Entrenador(cerebro, tasa=0.8)
        print("Entrenando XOR...")
        for epoch in range(5000):
            error = entrenador.entrenar_epoch(X, y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: error = {error:.6f}")
        print("\nResultados:")
        for inputs in X:
            pred = cerebro.predecir(inputs)
            print(f"{inputs} -> {pred[0][0]:.4f}")
    
    elif opcion == "2":
        from datos_reales import *
    elif opcion == "3":
        from entrenar_profundo import *
    elif opcion == "4":
        from entrenar_memoria import *
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
        print("   - red_neuronal.py: Cerebro básico (2 capas)")
        print("   - red_profunda.py: Cerebro profundo (5 capas + ReLU)")
        print("   - memoria.py: Cerebro con memoria circular")
        print("   - convolutional.py: Cerebro con visión")
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
