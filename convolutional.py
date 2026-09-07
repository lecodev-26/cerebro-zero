import numpy as np

class CapaConvolucional:
    def __init__(self, num_filtros, tamanio_filtro):
        self.num_filtros = num_filtros
        self.tamanio_filtro = tamanio_filtro
        self.filtros = np.random.randn(num_filtros, tamanio_filtro, tamanio_filtro) * 0.1
    
    def convolucionar(self, imagen):
        h, w = imagen.shape
        f = self.tamanio_filtro
        salida_h = h - f + 1
        salida_w = w - f + 1
        
        salidas = []
        for filtro in self.filtros:
            salida = np.zeros((salida_h, salida_w))
            for i in range(salida_h):
                for j in range(salida_w):
                    region = imagen[i:i+f, j:j+f]
                    salida[i, j] = np.sum(region * filtro)
            salida = np.maximum(0, salida)
            salidas.append(salida)
        
        return np.array(salidas)

class CerebroConVision:
    def __init__(self, tamanio_imagen=28):
        self.tamanio_imagen = tamanio_imagen
        self.conv1 = CapaConvolucional(8, 3)
        self.conv2 = CapaConvolucional(16, 3)
        
        salida1_h = tamanio_imagen - 3 + 1
        salida2_h = salida1_h - 3 + 1
        self.tam_final = salida2_h ** 2 * 16
        
        from red_neuronal import Cerebro
        self.cerebro = Cerebro([self.tam_final, 32, 10])
    
    def procesar(self, imagen):
        if len(imagen.shape) == 3:
            imagen = np.mean(imagen, axis=2)
        
        salida1 = self.conv1.convolucionar(imagen)
        
        salidas2 = []
        for i in range(salida1.shape[0]):
            salida2 = self.conv2.convolucionar(salida1[i])
            salidas2.append(salida2.flatten())
        
        vector = np.concatenate(salidas2)
        return self.cerebro.predecir(vector)
    
    def resumen(self):
        print("🧠 CEREBRO CON VISIÓN")
        print(f"   Tamaño imagen: {self.tamanio_imagen}x{self.tamanio_imagen}")
        print(f"   Conv1: 8 filtros 3x3")
        print(f"   Conv2: 16 filtros 3x3")
        print(f"   Capas densas: {self.cerebro.capas}")
        print(f"   Total parámetros: {self.cerebro.pesos[0].size + self.cerebro.pesos[1].size}")	

