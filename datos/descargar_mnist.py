import numpy as np
import gzip
import urllib.request
import os

def descargar_mnist():
    print("📥 DESCARGANDO MNIST...")
    
    # Crear carpeta
    os.makedirs('../modelos_guardados', exist_ok=True)
    
    urls = {
        'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    for name, url in urls.items():
        path = f'../modelos_guardados/{name}.gz'
        print(f"   Descargando {name} a {path}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"      ✅ {name} descargado")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print("✅ Descarga completada!")
    
    # Verificar archivos descargados
    print("\n📂 Verificando archivos:")
    for name in urls:
        path = f'../modelos_guardados/{name}.gz'
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   ✅ {name}.gz ({size:,} bytes)")
        else:
            print(f"   ❌ {name}.gz NO ENCONTRADO")
    
    # Función para leer MNIST
    def leer_mnist(imagenes_path, labels_path):
        if not os.path.exists(imagenes_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(f"No se encontraron archivos: {imagenes_path} o {labels_path}")
        
        with gzip.open(imagenes_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, rows * cols)
            data = data.astype(np.float32) / 255.0
        
        with gzip.open(labels_path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        return data, labels
    
    try:
        print("\n📂 Cargando datos...")
        X_train, y_train = leer_mnist(
            '../modelos_guardados/train_images.gz',
            '../modelos_guardados/train_labels.gz'
        )
        X_test, y_test = leer_mnist(
            '../modelos_guardados/test_images.gz',
            '../modelos_guardados/test_labels.gz'
        )
        
        np.save('../modelos_guardados/X_train.npy', X_train)
        np.save('../modelos_guardados/y_train.npy', y_train)
        np.save('../modelos_guardados/X_test.npy', X_test)
        np.save('../modelos_guardados/y_test.npy', y_test)
        
        print(f"\n✅ DATOS GUARDADOS!")
        print(f"   Entrenamiento: {X_train.shape[0]:,} imágenes")
        print(f"   Prueba: {X_test.shape[0]:,} imágenes")
        print(f"   Tamaño de imagen: {X_train.shape[1]} píxeles")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"\n❌ Error cargando datos: {e}")
        return None, None, None, None

if __name__ == "__main__":
    descargar_mnist()
