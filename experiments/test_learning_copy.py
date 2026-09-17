"""
Test: ¿Puede el Transformer aprender a copiar una secuencia?
Si SÍ → el backprop funciona y el problema es el dataset
Si NO → hay un bug real
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor import Tensor
from core.optimizers import Adam
from models.transformer import Transformer
from training.lm_trainer import LMTrainer


def test_learning():
    print("🧪 TEST: ¿Aprende el Transformer a COPIAR?")
    print("="*60)
    
    np.random.seed(42)
    
    # Dataset MUY simple: copiar una secuencia de 5 tokens
    # Tokens 0-9 (10 posibilidades)
    vocab_size = 10
    seq_len = 5
    num_samples = 200
    
    # Crear dataset
    X_data = np.random.randint(1, vocab_size, (num_samples, seq_len))
    y_data = X_data.copy()  # Target = mismo que input (tarea de copia)
    
    print(f"📊 Dataset:")
    print(f"   Vocab: {vocab_size}")
    print(f"   Seq len: {seq_len}")
    print(f"   Samples: {num_samples}")
    print(f"   Ejemplo X: {X_data[0]}")
    print(f"   Ejemplo y: {y_data[0]}")
    
    # Modelo MUY pequeño
    model = Transformer(
        vocab_size=vocab_size,
        d_model=32,      # Más grande que antes
        num_heads=2,
        d_ff=64,
        num_layers=2,    # 2 capas
        max_len=seq_len,
    )
    model.summary()
    
    # Trainer con lr más alto
    trainer = LMTrainer(model, learning_rate=0.05)
    
    # Entrenar con batches directamente
    print("\n🏋️ ENTRENANDO...\n")
    
    batch_size = 16
    epochs = 100
    batches_per_epoch = 5
    
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(batches_per_epoch):
            # Sample batch
            idx = np.random.choice(num_samples, batch_size, replace=False)
            X_batch = X_data[idx]
            y_batch = y_data[idx]
            
            loss = trainer.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss = {avg_loss:.4f}")
    
    print(f"\n📊 Resultados:")
    print(f"   Loss inicial: {losses[0]:.4f}")
    print(f"   Loss final:   {losses[-1]:.4f}")
    print(f"   Mejora:       {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    # Test de copia
    print(f"\n🔍 Test de copia:")
    test_X = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 1, 0, 1, 0]])
    logits = model.forward(Tensor(test_X)).data
    predictions = np.argmax(logits, axis=-1)
    
    for i in range(len(test_X)):
        print(f"   Input:  {test_X[i]}")
        print(f"   Output: {predictions[i]}")
        print(f"   ¿Correcto? {np.array_equal(test_X[i], predictions[i])}")
    
    return losses


if __name__ == "__main__":
    losses = test_learning()
    
    print("\n" + "="*60)
    if losses[-1] < losses[0] * 0.5:
        print("🎉 ¡El modelo APRENDE a copiar!")
        print("   El backprop funciona. El problema es el dataset/tamaño.")
    elif losses[-1] < losses[0] * 0.9:
        print("⚠️ El modelo aprende POCO.")
        print("   Puede ser el learning rate o el tamaño del modelo.")
    else:
        print("❌ El modelo NO aprende a copiar.")
        print("   Hay un BUG real en el backprop.")
    print("="*60)
