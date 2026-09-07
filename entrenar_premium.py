import numpy as np
from red_neuronal_premium import CerebroPremium

class EntrenadorPremium:
    def __init__(self, cerebro, tasa=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.cerebro = cerebro
        self.tasa = tasa
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Para Adam: momentos
        self.m_w = [np.zeros_like(w) for w in cerebro.pesos]
        self.v_w = [np.zeros_like(w) for w in cerebro.pesos]
        self.m_b = [np.zeros_like(b) for b in cerebro.sesgos]
        self.v_b = [np.zeros_like(b) for b in cerebro.sesgos]
        self.t = 0
    
    def mse(self, real, pred):
        return np.mean((real - pred) ** 2)
    
    def entrenar_epoch(self, X, y):
        # Forward pass
        salida = self.cerebro.forward(X, entrenando=True)
        
        # Calcular error (Entropía Cruzada)
        error = salida - y  # Para softmax + cross-entropy, el gradiente se simplifica
        grad = error
        
        # Backpropagation
        for i in range(len(self.cerebro.pesos)-1, -1, -1):
            # Gradiente de pesos y sesgos
            grad_w = np.dot(self.cerebro.activaciones[i].T, grad)
            grad_b = np.sum(grad, axis=0, keepdims=True)
            
            # Adam: actualización con momentos
            self.t += 1
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b ** 2)
            
            # Corrección de sesgo
            m_w_corr = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_corr = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_corr = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Actualizar pesos y sesgos
            self.cerebro.pesos[i] -= self.tasa * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
            self.cerebro.sesgos[i] -= self.tasa * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)
            
            # BatchNorm: actualizar gamma y beta (si está activado)
            if self.cerebro.use_batchnorm and i < len(self.cerebro.pesos)-1:
                # Gradiente para gamma y beta
                grad_bn = grad * self.cerebro.bn_z_norm[i]
                grad_gamma = np.sum(grad * self.cerebro.bn_z_norm[i], axis=0, keepdims=True)
                grad_beta = np.sum(grad, axis=0, keepdims=True)
                
                # Actualizar gamma y beta con Adam
                self.cerebro.bn_gamma[i] -= self.tasa * grad_gamma
                self.cerebro.bn_beta[i] -= self.tasa * grad_beta
            
            # Propagar gradiente a la capa anterior
            if i > 0:
                grad = np.dot(grad, self.cerebro.pesos[i].T)
                grad = grad * self.cerebro.relu_derivada(self.cerebro.lineales[i-1])
        
        # Calcular error (entropía cruzada)
        return self.cerebro.cross_entropy(y, salida)
    
    def entrenar(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            error = self.entrenar_epoch(X, y)
            if verbose and epoch % 100 == 0:
                print(f"   Epoch {epoch}: error = {error:.6f}")
        return error

def ejecutar_premium():
    print("🧠 CEREBRO PREMIUM CON TODAS LAS MEJORAS")
    print("="*50)
    print("   - Softmax (clasificación)")
    print("   - Entropía Cruzada (función de costo)")
    print("   - Batch Normalization (estabilidad)")
    print("   - Dropout (regularización)")
    print("   - Adam (optimizador)")
    print("="*50)
    
    # Crear cerebro: 2 entradas, 16 ocultas, 2 salidas (clasificación)
    cerebro = CerebroPremium(
        capas=[2, 16, 2],
        dropout_rate=0.2,
        use_batchnorm=True
    )
    cerebro.resumen()
    
    # Datos: XOR pero como clasificación (2 salidas)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])  # One-hot encoding
    
    # Repetir datos
    X_train = np.repeat(X, 100, axis=0)
    y_train = np.repeat(y, 100, axis=0)
    
    entrenador = EntrenadorPremium(cerebro, tasa=0.001)
    print("\n🏋️ ENTRENANDO...")
    entrenador.entrenar(X_train, y_train, epochs=2000)
    
    print("\n✅ RESULTADOS FINALES:")
    for inputs in X:
        pred = cerebro.predecir(inputs)
        clase = np.argmax(pred)
        print(f"   {inputs} → {pred} → Clase {clase}")
    
    cerebro.guardar("cerebro_premium.pkl")
    print("\n💾 Cerebro Premium guardado!")

if __name__ == "__main__":
    ejecutar_premium()
