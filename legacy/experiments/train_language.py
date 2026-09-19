"""
Entrenamiento de lenguaje con Transformer - Guardado corregido
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from language.tokenizer import Tokenizer
from core.transformer import Transformer

def generar_datos_entrenamiento(tokenizer, num_ejemplos=1000, max_len=10):
    frases = [
        "el cerebro zero aprende lenguaje",
        "la inteligencia artificial es fascinante",
        "los transformers son poderosos",
        "el aprendizaje profundo cambia el mundo",
        "las redes neuronales son increíbles",
        "el procesamiento de lenguaje natural avanza",
        "la memoria es clave para la ia",
        "el razonamiento es una habilidad fundamental",
        "las herramientas potencian la inteligencia",
        "el agente aprende de la experiencia"
    ]
    
    X = []
    y = []
    
    for _ in range(num_ejemplos):
        frase = np.random.choice(frases)
        tokens = tokenizer.encode_with_special(frase, add_bos=True, add_eos=True)
        
        for i in range(1, len(tokens)):
            X.append(tokens[:i])
            y.append(tokens[i])
    
    max_len = min(max_len, max(len(seq) for seq in X))
    
    X_padded = []
    y_padded = []
    
    for seq, target in zip(X, y):
        if len(seq) > max_len:
            seq = seq[-max_len:]
        else:
            seq = [0] * (max_len - len(seq)) + seq
        X_padded.append(seq)
        y_padded.append(target)
    
    return np.array(X_padded), np.array(y_padded)

def entrenar_transformer():
    print("🧠 ENTRENANDO TRANSFORMER")
    print("="*40)
    
    tokenizer = Tokenizer()
    try:
        tokenizer.load("vocab.json")
        print(f"📚 Vocabulario cargado: {tokenizer.vocab_size} palabras")
    except:
        print("❌ No se encontró vocab.json")
        print("   Ejecuta: python experiments/test_language.py")
        return
    
    print("\n📊 Generando datos...")
    X, y = generar_datos_entrenamiento(tokenizer, num_ejemplos=500, max_len=8)
    print(f"   Entradas: {X.shape}")
    print(f"   Salidas: {y.shape}")
    
    vocab_size = tokenizer.vocab_size
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=32,
        num_layers=2,
        max_len=10
    )
    
    print(f"\n🧠 Transformer creado:")
    print(f"   Vocabulario: {vocab_size}")
    
    print("\n🏋️ ENTRENANDO...")
    
    for epoch in range(5):
        total_loss = 0
        for i in range(len(X)):
            x = X[i:i+1]
            y_true = y[i]
            probs = transformer.forward(x)
            loss = -np.log(probs[0, -1, y_true] + 1e-10)
            total_loss += loss
        
        print(f"   Epoch {epoch+1}: loss = {total_loss/len(X):.4f}")
    
    print("\n✅ ENTRENAMIENTO COMPLETADO!")
    
    # Guardar solo pesos (no objetos con lambdas)
    pesos = {
        'embedding': transformer.embedding,
        'pos_encoding_encoding': transformer.pos_encoding.encoding,
        'W_out': transformer.W_out,
        'b_out': transformer.b_out,
        'layers': []
    }
    
    for layer in transformer.layers:
        pesos['layers'].append({
            'attention': {
                'W_q': layer.attention.W_q,
                'W_k': layer.attention.W_k,
                'W_v': layer.attention.W_v,
                'W_o': layer.attention.W_o
            },
            'ff': {
                'W1': layer.ff.W1,
                'b1': layer.ff.b1,
                'W2': layer.ff.W2,
                'b2': layer.ff.b2
            }
        })
    
    with open("../modelos_guardados/transformer_weights.pkl", 'wb') as f:
        pickle.dump(pesos, f)
    print("💾 Pesos guardados en modelos_guardados/transformer_weights.pkl")

if __name__ == "__main__":
    entrenar_transformer()
