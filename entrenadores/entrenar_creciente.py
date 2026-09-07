import sys
sys.path.append('..')
import numpy as np
import pickle
import os
from cerebros.red_neuronal import Cerebro

class CerebroProfundo(Cerebro):
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        for i in range(len(capas)-1):
            # Usar la misma inicialización que funciona
            w = np.random.randn(capas[i], capas[i+1]) * 0.1
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)

def ejecutar_creciente():
    archivo_estado = "../modelos_guardados/estado_creciente.pkl"
    archivo_cerebro = "../modelos_guardados/cerebro_creciente.pkl"
    
    # Usar semilla fija
    np.random.seed(42)
    
    if os.path.exists(archivo_estado):
        with open(archivo_estado, 'rb') as f:
            estado = pickle.load(f)
        capas = estado['capas']
        print(f"🧠 CARGANDO CEREBRO CON {len(capas)-1} CAPAS OCULTAS")
    else:
        capas = [2, 4, 1]  # Empezamos con 1 capa oculta
        print(f"🧠 CREANDO CEREBRO NUEVO CON 1 CAPA OCULTA")
    
    # Crecer: añadir una capa oculta
    nuevas_capas = capas.copy()
    if len(nuevas_capas) > 5:
        print("⚠️ LÍMITE MÁXIMO DE CAPAS ALCANZADO (5 capas ocultas)")
        nuevas_capas = capas
    else:
        # Añadir una capa oculta con 8 neuronas (que funciona)
        nuevas_capas.insert(-1, 8)
        print(f"🔥 CEREBRO CRECE: {len(capas)-1} → {len(nuevas_capas)-1} CAPAS OCULTAS")
    
    print("="*50)
    print(f"   Capas: {nuevas_capas}")
    
    cerebro = CerebroProfundo(nuevas_capas)
    cerebro.resumen()
    
    # Datos XOR
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
        pickle.dump({'capas': nuevas_capas}, f)
    
    print(f"\n💾 CEREBRO GUARDADO CON {len(nuevas_capas)-1} CAPAS OCULTAS")
    print(f"   La próxima vez que ejecutes, crecerá a {len(nuevas_capas)} capas ocultas")

if __name__ == "__main__":
    ejecutar_creciente()
