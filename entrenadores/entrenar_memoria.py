import sys
sys.path.append('..')
import numpy as np
from cerebros.red_neuronal import Cerebro

def ejecutar_memoria():
    print("🧠 EJEMPLO DE MEMORIA CON CEREBRO BÁSICO")
    print("="*50)
    
    try:
        cerebro = Cerebro([2, 8, 1])
        cerebro.cargar("../modelos_guardados/xor_cerebro_mejorado.pkl")
        
        print("\n📊 DEMOSTRACIÓN DE MEMORIA:")
        print("   Usamos el cerebro entrenado en XOR.")
        print("   Simulamos una secuencia de entradas.")
        
        secuencia = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0]]
        
        print("\n🔮 Predicciones:")
        for i, entrada in enumerate(secuencia):
            pred = cerebro.predecir(entrada)
            print(f"   Paso {i+1}: {entrada} → {pred[0][0]:.4f}")
        
        print("\n💡 Este cerebro fue entrenado con XOR.")
        print("   Para memoria real, se necesitan datos secuenciales.")
        
    except FileNotFoundError:
        print("❌ No se encontró el archivo xor_cerebro_mejorado.pkl")
        print("   Primero ejecuta la opción 1 para entrenar el cerebro.")

if __name__ == "__main__":
    ejecutar_memoria()
