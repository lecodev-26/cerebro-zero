import numpy as np
import pickle

class CerebroPremium:
    def __init__(self, capas, dropout_rate=0.0, use_batchnorm=False):
        """
        capas: [entrada, oculta1, oculta2, ..., salida]
        dropout_rate: 0.0 a 0.5 (fracción de neuronas a apagar)
        use_batchnorm: True/False (normalizar capas ocultas)
        """
        self.capas = capas
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.pesos = []
        self.sesgos = []
        
        # Inicialización He (para ReLU)
        for i in range(len(capas)-1):
            w = np.random.randn(capas[i], capas[i+1]) * np.sqrt(2.0 / capas[i])
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(w)
            self.sesgos.append(b)
        
        # Si usamos BatchNorm, necesitamos parámetros por capa
        if use_batchnorm:
            self.bn_gamma = [np.ones((1, capas[i+1])) for i in range(len(capas)-1)]
            self.bn_beta = [np.zeros((1, capas[i+1])) for i in range(len(capas)-1)]
            self.bn_mean = [np.zeros((1, capas[i+1])) for i in range(len(capas)-1)]
            self.bn_var = [np.ones((1, capas[i+1])) for i in range(len(capas)-1)]
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivada(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x):
        """Softmax para clasificación multiclase"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy(self, y_real, y_pred):
        """Entropía Cruzada (evita log(0))"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_real * np.log(y_pred), axis=1))
    
    def mse(self, y_real, y_pred):
        return np.mean((y_real - y_pred) ** 2)
    
    def forward(self, X, entrenando=False):
        self.activaciones = [X]
        self.lineales = []
        self.dropout_masks = []
        self.bn_z_norm = []
        
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            # Linear: z = W*A + b
            z = np.dot(self.activaciones[-1], w) + b
            self.lineales.append(z)
            
            # Batch Normalization (si está activado y es capa oculta)
            if self.use_batchnorm and i < len(self.pesos) - 1:
                if entrenando:
                    # Calcular media y varianza de este batch
                    mean = np.mean(z, axis=0, keepdims=True)
                    var = np.var(z, axis=0, keepdims=True)
                    self.bn_mean[i] = mean
                    self.bn_var[i] = var
                    
                    # Normalizar
                    z_norm = (z - mean) / np.sqrt(var + 1e-8)
                    # Escalar y desplazar
                    z = self.bn_gamma[i] * z_norm + self.bn_beta[i]
                    self.bn_z_norm.append(z_norm)
                else:
                    # Usar medias acumuladas (para inferencia)
                    z_norm = (z - self.bn_mean[i]) / np.sqrt(self.bn_var[i] + 1e-8)
                    z = self.bn_gamma[i] * z_norm + self.bn_beta[i]
                    self.bn_z_norm.append(z_norm)
            
            # Activación: ReLU para ocultas, Softmax para salida (clasificación)
            if i == len(self.pesos) - 1:
                a = self.softmax(z)  # Salida: probabilidades
            else:
                a = self.relu(z)  # Ocultas: ReLU
                # Dropout (solo en entrenamiento)
                if entrenando and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                    a = a * mask / (1 - self.dropout_rate)  # Escalar para mantener magnitud
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(np.ones_like(a))
            
            self.activaciones.append(a)
        
        return self.activaciones[-1]
    
    def predecir(self, X):
        """Predicción (sin dropout, sin batchnorm de entrenamiento)"""
        return self.forward(X, entrenando=False)
    
    def guardar(self, nombre="cerebro_premium.pkl"):
        with open(nombre, 'wb') as f:
            data = {
                'capas': self.capas,
                'pesos': self.pesos,
                'sesgos': self.sesgos,
                'dropout_rate': self.dropout_rate,
                'use_batchnorm': self.use_batchnorm,
            }
            if self.use_batchnorm:
                data['bn_gamma'] = self.bn_gamma
                data['bn_beta'] = self.bn_beta
                data['bn_mean'] = self.bn_mean
                data['bn_var'] = self.bn_var
            pickle.dump(data, f)
        print(f"💾 Cerebro Premium guardado en {nombre}")
    
    def cargar(self, nombre="cerebro_premium.pkl"):
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.capas = data['capas']
        self.pesos = data['pesos']
        self.sesgos = data['sesgos']
        self.dropout_rate = data.get('dropout_rate', 0.0)
        self.use_batchnorm = data.get('use_batchnorm', False)
        if self.use_batchnorm:
            self.bn_gamma = data['bn_gamma']
            self.bn_beta = data['bn_beta']
            self.bn_mean = data['bn_mean']
            self.bn_var = data['bn_var']
        print(f"📂 Cerebro Premium cargado desde {nombre}")
    
    def resumen(self):
        print("🧠 CEREBRO PREMIUM (CON TODAS LAS MEJORAS)")
        print("="*50)
        print(f"   Capas: {self.capas}")
        print(f"   Dropout: {self.dropout_rate*100:.0f}%")
        print(f"   BatchNorm: {'✅' if self.use_batchnorm else '❌'}")
        total_params = sum(w.size + b.size for w, b in zip(self.pesos, self.sesgos))
        print(f"   Parámetros totales: {total_params}")
        for i, (w, b) in enumerate(zip(self.pesos, self.sesgos)):
            tipo = "ReLU" if i < len(self.pesos)-1 else "Softmax"
            print(f"   Capa {i+1}: {w.shape} → {b.shape} (activación: {tipo})")
