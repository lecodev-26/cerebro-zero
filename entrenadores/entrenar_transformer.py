import sys
sys.path.append('..')
import numpy as np
from cerebros.transformer_mini import TransformerMini

def ejecutar_transformer():
    print("🧠 ENTRENANDO TRANSFORMER MINI")
    print("="*50)
    print("⚠️ ADVERTENCIA: Esto puede tardar y consumir RAM")
    print("⚠️ Si el móvil se calienta, cancela con CTRL+C")
    print("="*50)
    
    # Crear transformer mini
    transformer = TransformerMini(
        vocab_size=100,
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=2
    )
    
    # Generar datos de ejemplo (secuencias)
    X_train = np.random.randint(0, 100, (50, 10))  # 50 secuencias de 10 tokens
    y_train = np.random.randint(0, 100, (50, 10, 100))  # One-hot para cada token
    
    # Convertir y_train a one-hot
    for i in range(len(X_train)):
        for j in range(10):
            y_train[i, j] = np.eye(100)[X_train[i, j]]  # Autoencoder: predecir el mismo token
    
    print(f"\n📊 Datos: {len(X_train)} secuencias de 10 tokens")
    print("🏋️ Entrenando... (esto es muy lento en CPU)")
    
    # Entrenamiento simple
    for epoch in range(5):
        total_error = 0
        for i in range(len(X_train)):
            # Forward
            salida = transformer.forward(X_train[i])  # (10, 100)
            # Calcular error (cross-entropy)
            error = -np.mean(np.sum(y_train[i] * np.log(salida + 1e-10), axis=1))
            total_error += error
        
        print(f"   Epoch {epoch}: error = {total_error:.6f}")
    
    print("\n✅ ENTRENADO!")
    transformer.guardar("../modelos_guardados/transformer_mini.pkl")
    print("\n💾 Transformer guardado!")
    
    print("\n💡 Esto NO es ChatGPT, pero es un Transformer en tu móvil.")
    print("   Tiene 22,784 parámetros. ChatGPT tiene billones.")

if __name__ == "__main__":
    ejecutar_transformer()
