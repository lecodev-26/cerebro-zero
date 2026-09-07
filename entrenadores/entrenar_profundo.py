import sys
sys.path.append('..')
import numpy as np
from cerebros.red_neuronal import Cerebro

def ejecutar_profundo():
    print("🧠 CARGANDO CEREBRO GUARDADO (XOR)")
    print("="*40)
    
    try:
        cerebro = Cerebro([2, 8, 1])
        cerebro.cargar("../modelos_guardados/xor_cerebro_mejorado.pkl")
        
        print("\n✅ CEREBRO CARGADO EXITOSAMENTE!")
        print("\n📊 PREDICCIONES:")
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        for inputs in X:
            pred = cerebro.predecir(inputs)
            print(f"   {inputs} → {pred[0][0]:.4f}")
        
        print("\n💡 Este cerebro ya fue entrenado con XOR.")
        print("   Para reentrenar, usa la opción 1.")
        
    except FileNotFoundError:
        print("❌ No se encontró el archivo xor_cerebro_mejorado.pkl")
        print("   Primero ejecuta la opción 1 para entrenar el cerebro.")

if __name__ == "__main__":
    ejecutar_profundo()
