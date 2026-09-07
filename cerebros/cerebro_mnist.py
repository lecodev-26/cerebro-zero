import sys
sys.path.append('..')
import numpy as np
import pickle

class CerebroMNIST:
    def __init__(self, capas, tasa=0.01, dropout_rate=0.1):
        self.capas = capas
        self.tasa = tasa
        self.dropout_rate = dropout_rate
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas)-1):
            lim = np.sqrt(6.0 / (capas[i] + capas[i+1]))
            w = np.random.uniform(-lim, lim, (capas[i], capas[i+1]))
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def relu(self, x):
        return np.maximum(0, np.clip(x, -10, 10))
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        x = np.clip(x, -10, 10)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
    
    def forward(self, X, entrenando=False):
        self.activaciones = [X]
        self.lineales = []
        self.dropout_masks = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            z = np.clip(z, -10, 10)
            self.lineales.append(z)
            
            if i == len(self.pesos) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
                # Dropout solo en entrenamiento
                if entrenando and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                    a = a * mask / (1 - self.dropout_rate)
                    self.dropout_masks.append(mask)
            
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def backward(self, X, y_onehot, salida):
        grad = salida - y_onehot
        grad = np.clip(grad, -1, 1)
        
        for i in range(len(self.pesos)-1, -1, -1):
            self.pesos[i] -= self.tasa * np.dot(self.activaciones[i].T, grad)
            self.sesgos[i] -= self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.pesos[i].T)
                grad = grad * self.relu_derivada(self.lineales[i-1])
                grad = np.clip(grad, -1, 1)
        
        return np.mean((salida - y_onehot) ** 2)
    
    def entrenar(self, X, y_onehot, epochs=10, batch_size=128):
        print(f"🏋️ ENTRENANDO CON {X.shape[0]:,} EJEMPLOS...")
        n = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            total_error = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                salida = self.forward(X_batch, entrenando=True)
                error = self.backward(X_batch, y_batch, salida)
                total_error += error
            
            avg_error = total_error / max(1, n // batch_size)
            print(f"   Epoch {epoch+1}/{epochs}: error = {avg_error:.6f}")
    
    def predecir(self, X):
        return self.forward(X, entrenando=False)
    
    def precision(self, X, y):
        pred = self.forward(X, entrenando=False)
        pred_clases = np.argmax(pred, axis=1)
        return np.mean(pred_clases == y)
    
    def guardar(self, nombre="cerebro_mnist.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos,
                'sesgos': self.sesgos,
                'capas': self.capas
            }, f)
        print(f"💾 Cerebro MNIST guardado en {nombre}")
    
    def resumen(self):
        print("🧠 CEREBRO MNIST (ReLU + Softmax + Dropout)")
        print("="*40)
        print(f"   Capas: {self.capas}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params:,}")
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            tipo = "ReLU" if i < len(self.pesos)-1 else "Softmax"
            print(f"   Capa {i+1}: {w.shape} → {b.shape} ({tipo})")
        print(f"   Dropout: {self.dropout_rate*100:.0f}%")
