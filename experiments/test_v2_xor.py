import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.tensor_v2 import Tensor
from core.layers import Linear, Sequential
from core.activations import ReLU, Sigmoid
from core.losses import BCELoss

print("🧠 PROBANDO AUTOGRAD V2 CON XOR")
print("="*40)

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
Y = np.array([[0],[1],[1],[0]], dtype=np.float64)

x_t = Tensor(X)
y_t = Tensor(Y)

model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid()
)

print(f"Modelo:\n{model}")

loss_fn = BCELoss()
lr = 0.01

print("\n🏋️ ENTRENANDO XOR CON AUTOGRAD V2...")
for epoch in range(10000):
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    
    for param in model.parameters():
        if param.grad is not None:
            param.data -= lr * param.grad
    
    model.zero_grad()
    
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: loss = {loss.data:.6f}")

print("\n✅ RESULTADOS FINALES:")
for i, x in enumerate(X):
    pred = model(Tensor(x)).data
    print(f"   {x} → {pred[0,0]:.4f}")
