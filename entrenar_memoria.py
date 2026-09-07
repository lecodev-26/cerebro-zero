import numpy as np
from red_neuronal import Cerebro

def ejecutar_memoria():
    print("🧠 EJEMPLO DE MEMORIA CON CEREBRO BÁSICO")
    print("="*50)
    
    # Crear cerebro básico (ya entrenado)
    cerebro = Cerebro([2, 8, 1])
    cerebro.cargar("xor_cerebro_mejorado.pkl")
    
    print("\n📊 DEMOSTRACIÓN DE MEMORIA:")
    print("   Usamos el cerebro entrenado en XOR.")
    print("   Simulamos una secuencia de entradas.")
    
    # Secuencia de ejemplo
    secuencia = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]]
    
    print("\n🔮 Predicciones:")
    for i, entrada in enumerate(secuencia):
        pred = cerebro.predecir(entrada)
        print(f"   Paso {i+1}: {entrada} → {pred[0][0]:.4f}")
    
    print("\n💡 Este cerebro fue entrenado con XOR.")
    print("   Para memoria real, se necesitan datos secuenciales.")
    
if __name__ == "__main__":
    ejecutar_memoria()
