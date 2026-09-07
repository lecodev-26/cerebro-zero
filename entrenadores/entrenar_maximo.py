import sys
sys.path.append('..')
import numpy as np
from cerebros.cerebro_gigante import CerebroGigante

def ejecutar_maximo():
    print("🔥 CEREBRO MÁXIMO PARA TU MÓVIL")
    print("="*50)
    print("Vamos a usar 512 neuronas en una capa oculta.")
    print("Tu móvil tiene 3.7GB de RAM. Esto debería funcionar.")
    print("="*50)
    
    # Crear cerebro masivo: 10 entradas, 512 ocultas, 1 salida
    # 10*512 + 512*1 = 5,632 parámetros (¡más que el anterior!)
    cerebro = CerebroGigante([10, 512, 1])
    
    # Generar datos más complejos
    X = np.random.randn(2000, 10) * 2
    y = np.sin(X.sum(axis=1, keepdims=True)) + np.random.randn(2000, 1) * 0.1
    
    # Escalar datos
    X = X / 5.0
    y = y / 2.0
    
    from entrenadores.entrenar_gigante import EntrenadorGigante
    entrenador = EntrenadorGigante(cerebro, tasa=0.0005)
    
    print(f"\n📊 Datos: {len(X)} ejemplos con 10 características")
    print("🏋️ Entrenando... (esto puede tomar varios minutos)")
    
    entrenador.entrenar(X, y, epochs=200)
    
    print("\n✅ ENTRENADO!")
    print("\n📈 Predicciones:")
    for i in range(5):
        pred = cerebro.predecir(X[i:i+1])
        print(f"   Ejemplo {i+1}: {pred[0][0]:.3f} (real: {y[i][0]:.3f})")
    
    cerebro.guardar("../modelos_guardados/cerebro_maximo.pkl")
    print("\n💾 Cerebro máximo guardado!")
