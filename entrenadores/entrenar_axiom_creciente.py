import sys
sys.path.append('..')
import numpy as np
import pickle
import os
from cerebros.red_neuronal import Cerebro

def ejecutar_axiom_creciente():
    archivo_estado = "../modelos_guardados/estado_axiom.pkl"
    archivo_cerebro = "../modelos_guardados/cerebro_axiom.pkl"
    
    # Usar semilla fija para reproducibilidad
    np.random.seed(42)
    
    if os.path.exists(archivo_estado):
        with open(archivo_estado, 'rb') as f:
            estado = pickle.load(f)
        neuronas = estado['neuronas']
        print(f"🧠 CARGANDO CEREBRO DE {neuronas} NEURONAS")
    else:
        neuronas = 4
        print(f"🧠 CREANDO CEREBRO NUEVO CON {neuronas} NEURONAS")
    
    nuevas_neuronas = neuronas * 2
    if nuevas_neuronas > 1024:
        nuevas_neuronas = 1024
        print("⚠️ LÍMITE MÁXIMO ALCANZADO (1024 neuronas)")
    
    print(f"🔥 CEREBRO VA A CRECER DE {neuronas} A {nuevas_neuronas} NEURONAS")
    print("="*50)
    
    # Usar el cerebro básico (el que sabemos que funciona)
    cerebro = Cerebro([2, nuevas_neuronas, 1])
    cerebro.resumen()
    
    # Datos XOR (con más ejemplos)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    X_train = np.repeat(X, 200, axis=0)
    y_train = np.repeat(y, 200, axis=0)
    
    from entrenadores.entrenar_xor import Entrenador
    entrenador = Entrenador(cerebro, tasa=0.8)
    
    print("\n🏋️ ENTRENANDO...")
    for epoch in range(20000):
        error = entrenador.entrenar_epoch(X_train, y_train)
        if epoch % 2000 == 0:
            print(f"   Epoch {epoch}: error = {error:.6f}")
    
    print("\n✅ RESULTADOS:")
    for inputs in X:
        pred = cerebro.predecir(inputs)
        print(f"   {inputs} → {pred[0][0]:.4f}")
    
    cerebro.guardar(archivo_cerebro)
    
    with open(archivo_estado, 'wb') as f:
        pickle.dump({'neuronas': nuevas_neuronas}, f)
    
    print(f"\n💾 CEREBRO GUARDADO CON {nuevas_neuronas} NEURONAS")
    print(f"   La próxima vez que ejecutes, crecerá a {nuevas_neuronas * 2} neuronas")

if __name__ == "__main__":
    ejecutar_axiom_creciente()
