import sys
sys.path.append('..')
import numpy as np

class Autoencoder:
    def __init__(self, entrada_size, codificado_size=3):
        self.entrada_size = entrada_size
        self.codificado_size = codificado_size
        self.media = None
        self.componentes = None
    
    def entrenar(self, X, epochs=500):
        """Entrena usando PCA (equivalente a autoencoder lineal)"""
        print(f"📦 ENTRENANDO AUTOENCODER (PCA)")
        print(f"   {self.entrada_size} → {self.codificado_size} → {self.entrada_size}")
        
        # Centrar datos
        self.media = np.mean(X, axis=0, keepdims=True)
        X_centrado = X - self.media
        
        # Calcular matriz de covarianza
        cov = np.cov(X_centrado.T)
        
        # Calcular autovectores y autovalores
        eigenvals, eigenvectores = np.linalg.eigh(cov)
        
        # Ordenar por autovalor descendente
        idx = np.argsort(eigenvals)[::-1]
        eigenvectores = eigenvectores[:, idx]
        
        # Seleccionar los primeros 'codificado_size' componentes
        self.componentes = eigenvectores[:, :self.codificado_size]
        
        # Calcular error de reconstrucción
        X_reconstruido = self.reconstruir(X)
        error = np.mean((X - X_reconstruido) ** 2)
        print(f"   Error de reconstrucción: {error:.6f}")
        return error
    
    def codificar(self, X):
        """Comprime datos a tamaño reducido"""
        X_centrado = X - self.media
        return np.dot(X_centrado, self.componentes)
    
    def decodificar(self, codificado):
        """Reconstruye desde compresión"""
        return self.media + np.dot(codificado, self.componentes.T)
    
    def reconstruir(self, X):
        """Codifica y decodifica (reconstrucción)"""
        codificado = self.codificar(X)
        return self.decodificar(codificado)
    
    def guardar(self, nombre="autoencoder.pkl"):
        import pickle
        with open(nombre, 'wb') as f:
            pickle.dump({
                'media': self.media,
                'componentes': self.componentes,
                'entrada_size': self.entrada_size,
                'codificado_size': self.codificado_size
            }, f)
        print(f"💾 Autoencoder guardado en {nombre}")
    
    def cargar(self, nombre="autoencoder.pkl"):
        import pickle
        with open(nombre, 'rb') as f:
            data = pickle.load(f)
        self.media = data['media']
        self.componentes = data['componentes']
        self.entrada_size = data['entrada_size']
        self.codificado_size = data['codificado_size']
        print(f"📂 Autoencoder cargado desde {nombre}")

def ejecutar_autoencoder():
    print("🧠 AUTOENCODER (PCA)")
    print("="*40)
    print("   El autoencoder comprime datos y los reconstruye.")
    print("   Usamos PCA, que es equivalente a un autoencoder lineal.")
    print("="*40)
    
    # Generar datos aleatorios con estructura
    np.random.seed(42)
    X = np.random.randn(200, 10)
    # Añadir correlación (para que tenga estructura)
    X = np.dot(X, np.random.randn(10, 10))
    
    ae = Autoencoder(entrada_size=10, codificado_size=3)
    ae.entrenar(X, epochs=500)
    
    # Probar codificación y decodificación
    muestra = X[:5]
    codificado = ae.codificar(muestra)
    reconstruido = ae.reconstruir(muestra)
    
    print("\n📊 DEMOSTRACIÓN:")
    print("   Original (5 ejemplos, 10 dimensiones):")
    print(f"   {muestra.shape}")
    print("\n   Codificado (5 ejemplos, 3 dimensiones):")
    print(f"   {codificado.shape}")
    print("\n   Reconstruido (5 ejemplos, 10 dimensiones):")
    print(f"   {reconstruido.shape}")
    print("\n✅ Autoencoder funcionando!")
    
    ae.guardar("autoencoder.pkl")

if __name__ == "__main__":
    ejecutar_autoencoder()
