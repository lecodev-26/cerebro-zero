import numpy as np
import json

class GestorDatos:
    @staticmethod
    def generar_xor(n=100):
        """Genera datos XOR con ruido"""
        X = np.random.randint(0, 2, (n, 2))
        y = np.array([[x[0] ^ x[1]] for x in X])
        return X.astype(float), y.astype(float)
    
    @staticmethod
    def generar_seno(n=200, ruido=0.1):
        """Genera datos de función seno"""
        X = np.random.rand(n, 1) * 10
        y = np.sin(X) + np.random.randn(n, 1) * ruido
        return X, y
    
    @staticmethod
    def generar_espiral(n=300, clases=2):
        """Genera datos en forma de espiral (clasificación)"""
        t = np.linspace(0, 4*np.pi, n)
        r = t / (2*np.pi)
        X = np.zeros((n, 2))
        y = np.zeros((n, 1))
        
        for i in range(n):
            if i < n//2:
                X[i] = [r[i]*np.cos(t[i]), r[i]*np.sin(t[i])]
                y[i] = 0
            else:
                X[i] = [r[i]*np.cos(t[i] + np.pi), r[i]*np.sin(t[i] + np.pi)]
                y[i] = 1
        
        # Añadir ruido
        X += np.random.randn(n, 2) * 0.05
        return X, y
    
    @staticmethod
    def guardar_csv(X, y, nombre="datos.csv"):
        """Guarda datos en CSV"""
        data = np.hstack([X, y])
        np.savetxt(nombre, data, delimiter=",", header="entradas,salida", comments="")
        print(f"💾 Datos guardados en {nombre}")
    
    @staticmethod
    def cargar_csv(nombre="datos.csv"):
        """Carga datos desde CSV"""
        data = np.loadtxt(nombre, delimiter=",", skiprows=1)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        print(f"📂 Datos cargados desde {nombre}")
        return X, y
