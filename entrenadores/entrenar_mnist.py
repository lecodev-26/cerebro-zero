import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle

class CerebroMNISTGigante:
    def __init__(self, capas, tasa=0.1, dropout=0.2):
        self.capas = capas
        self.tasa = tasa
        self.dropout_rate = dropout
        self.pesos = []
        self.sesgos = []
        
        print(f"🧠 CREANDO CEREBRO GIGANTE ({capas[1]} neuronas ocultas)...")
        
        for i in range(len(capas)-1):
            # Inicialización He para redes profundas
            w = np.random.randn(capas[i], capas[i+1]) * np.sqrt(2.0 / capas[i])
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        # Evitar overflow
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)
    
    def forward(self, X, entrenando=False):
        self.activaciones = [X]
        self.lineales = []
        self.mascaras = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            z = np.dot(self.activaciones[-1], w) + b
            self.lineales.append(z)
            
            if i == len(self.pesos) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
                # Dropout
                if entrenando and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                    a = a * mask / (1 - self.dropout_rate)
                    self.mascaras.append(mask)
            
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def backward(self, X, y_onehot, salida):
        # Gradiente para softmax + cross-entropy
        grad = salida - y_onehot
        grad = np.clip(grad, -1, 1)  # Clip para evitar explosión
        
        for i in range(len(self.pesos)-1, -1, -1):
            self.pesos[i] -= self.tasa * np.dot(self.activaciones[i].T, grad)
            self.sesgos[i] -= self.tasa * np.sum(grad, axis=0, keepdims=True)
            
            if i > 0:
                grad = np.dot(grad, self.pesos[i].T)
                grad = grad * self.relu_derivada(self.lineales[i-1])
                grad = np.clip(grad, -1, 1)
        
        return np.mean((salida - y_onehot) ** 2)
    
    def entrenar(self, X, y_onehot, epochs=50, batch_size=128):
        print(f"🏋️ ENTRENANDO CON {X.shape[0]:,} EJEMPLOS...")
        print(f"   Dropout: {self.dropout_rate*100:.0f}%")
        print(f"   Batch size: {batch_size}")
        print("")
        
        n = X.shape[0]
        for epoch in range(epochs):
            # Mezclar datos
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
            
            if epoch % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: error = {total_error/(n//batch_size):.6f}")
    
    def predecir(self, X):
        return self.forward(X, entrenando=False)
    
    def precision(self, X, y):
        pred = self.forward(X, entrenando=False)
        return np.mean(np.argmax(pred, axis=1) == y)
    
    def guardar(self, nombre="cerebro_mnist_gigante.pkl"):
        with open(nombre, 'wb') as f:
            pickle.dump({
                'pesos': self.pesos,
                'sesgos': self.sesgos,
                'capas': self.capas
            }, f)
        print(f"💾 Cerebro MNIST Gigante guardado en {nombre}")
    
    def resumen(self):
        print("🧠 CEREBRO MNIST GIGANTE")
        print("="*40)
        print(f"   Capas: {self.capas}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params:,}")
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            tipo = "ReLU" if i < len(self.pesos)-1 else "Softmax"
            print(f"   Capa {i+1}: {w.shape} → {b.shape} ({tipo})")
        print(f"   Dropout: {self.dropout_rate*100:.0f}%")
        print(f"   Tasa de aprendizaje: {self.tasa}")
        print("="*40)

def ejecutar_mnist():
    print("🔥 CEREBRO MNIST GIGANTE - HASTA QUE PETE")
    print("="*50)
    print("⚠️ ESTO VA A CALENTAR TU MÓVIL")
    print("⚠️ Puede tardar 30-60 minutos")
    print("⚠️ Si ves que se calienta, cancela con CTRL+C")
    print("="*50)
    
    # Cargar datos
    try:
        X_train = np.load('../modelos_guardados/X_train.npy')
        y_train = np.load('../modelos_guardados/y_train.npy')
        X_test = np.load('../modelos_guardados/X_test.npy')
        y_test = np.load('../modelos_guardados/y_test.npy')
        print(f"\n✅ Datos cargados correctamente!")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return
    
    print(f"\n📊 DATOS CARGADOS:")
    print(f"   Entrenamiento: {X_train.shape[0]:,} imágenes")
    print(f"   Prueba: {X_test.shape[0]:,} imágenes")
    
    # Usar 30,000 ejemplos (mitad del dataset para no petar)
    n_ejemplos = 30000
    X_train_small = X_train[:n_ejemplos]
    y_train_small = y_train[:n_ejemplos]
    
    # One-hot encoding
    y_onehot = np.zeros((len(y_train_small), 10))
    y_onehot[np.arange(len(y_train_small)), y_train_small] = 1
    
    # Cerebro: 784 → 1024 → 10
    cerebro = CerebroMNISTGigante([784, 1024, 10], tasa=0.05, dropout=0.2)
    cerebro.resumen()
    
    print(f"\n🏋️ ENTRENANDO CON {n_ejemplos:,} EJEMPLOS...")
    print("   Esto puede tardar 20-40 minutos")
    print("   ¡Pon el móvil a cargar!")
    
    cerebro.entrenar(X_train_small, y_onehot, epochs=30, batch_size=128)
    
    # Evaluar
    print("\n📊 EVALUANDO...")
    prec_train = cerebro.precision(X_train_small, y_train_small)
    print(f"   Precisión entrenamiento: {prec_train*100:.2f}%")
    
    prec_test = cerebro.precision(X_test[:2000], y_test[:2000])
    print(f"   Precisión prueba: {prec_test*100:.2f}%")
    
    cerebro.guardar("../modelos_guardados/cerebro_mnist_gigante.pkl")
    print("\n💾 Cerebro MNIST Gigante guardado!")

if __name__ == "__main__":
    ejecutar_mnist()
