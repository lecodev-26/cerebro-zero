"""
Cargador de datasets para entrenamiento de lenguaje.
Tokeniza el texto y genera batches de (X, y) con ventana deslizante.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Tuple, Optional


class TextDataset:
    """
    Dataset de texto tokenizado.
    
    Uso:
        ds = TextDataset(path="datasets/train.txt", tokenizer=tok, context_len=16)
        X, y = ds.get_batch(batch_size=4)
    """
    
    def __init__(self, path: str, tokenizer, context_len: int = 16,
                 add_bos_eos: bool = True):
        """
        Args:
            path: ruta al archivo de texto
            tokenizer: instancia del Tokenizer
            context_len: longitud del contexto (ventana)
            add_bos_eos: si añadir <BOS> y <EOS> al inicio/final
        """
        self.path = path
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.add_bos_eos = add_bos_eos
        
        # Cargar y tokenizar
        self.tokens = self._load_and_tokenize()
    
    def _load_and_tokenize(self) -> List[int]:
        """Carga el archivo y lo tokeniza"""
        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenizar por línea
        lineas = [line.strip() for line in text.split('\n') if line.strip()]
        
        tokens = []
        if self.add_bos_eos:
            tokens.append(self.tokenizer.vocab.get('<BOS>', 2))
        
        for linea in lineas:
            ids = self.tokenizer.encode(linea)
            tokens.extend(ids)
            if self.add_bos_eos:
                tokens.append(self.tokenizer.vocab.get('<SEP>', 4))
        
        if self.add_bos_eos:
            tokens.append(self.tokenizer.vocab.get('<EOS>', 3))
        
        return tokens
    
    def __len__(self):
        """Número de muestras disponibles"""
        if len(self.tokens) <= self.context_len:
            return 0
        return len(self.tokens) - self.context_len
    
    def get_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (X, y) donde:
        - X = tokens[idx : idx + context_len]
        - y = tokens[idx + 1 : idx + context_len + 1]
        """
        X = np.array(self.tokens[idx:idx + self.context_len], dtype=np.int64)
        y = np.array(self.tokens[idx + 1:idx + self.context_len + 1], dtype=np.int64)
        return X, y
    
    def get_batch(self, batch_size: int = 4, shuffle: bool = True):
        """
        Devuelve un batch aleatorio de (X, y).
        X: (batch_size, context_len)
        y: (batch_size, context_len)
        """
        n = len(self)
        if n == 0:
            raise ValueError("Dataset demasiado pequeño")
        
        if shuffle:
            indices = np.random.randint(0, n, size=batch_size)
        else:
            indices = np.arange(batch_size) % n
        
        X_batch = []
        y_batch = []
        for i in indices:
            X, y = self.get_sample(i)
            X_batch.append(X)
            y_batch.append(y)
        
        return np.stack(X_batch), np.stack(y_batch)
    
    def all_tokens(self) -> np.ndarray:
        """Devuelve todos los tokens como array"""
        return np.array(self.tokens, dtype=np.int64)
    
    def stats(self) -> dict:
        """Estadísticas del dataset"""
        return {
            'path': self.path,
            'num_tokens': len(self.tokens),
            'num_samples': len(self),
            'context_len': self.context_len,
        }
    
    def __repr__(self):
        return f"TextDataset({os.path.basename(self.path)}, tokens={len(self.tokens)}, samples={len(self)})"


class DatasetCollection:
    """
    Colección de datasets: train, validation, test.
    """
    
    def __init__(self, datasets_dir: str = "datasets", tokenizer=None,
                 context_len: int = 16):
        self.datasets_dir = datasets_dir
        self.tokenizer = tokenizer
        self.context_len = context_len
        
        self.train = None
        self.validation = None
        self.test = None
        
        if tokenizer is not None:
            self._load_all()
    
    def _load_all(self):
        """Carga los 3 datasets"""
        for split in ['train', 'validation', 'test']:
            path = os.path.join(self.datasets_dir, f"{split}.txt")
            if os.path.exists(path):
                ds = TextDataset(path, self.tokenizer, self.context_len)
                setattr(self, split, ds)
    
    def summary(self):
        """Resumen de los datasets"""
        print("📊 DATASETS")
        print("="*50)
        for name in ['train', 'validation', 'test']:
            ds = getattr(self, name)
            if ds is not None:
                s = ds.stats()
                print(f"\n{name.upper()}:")
                print(f"   Tokens: {s['num_tokens']:,}")
                print(f"   Muestras: {s['num_samples']:,}")
                print(f"   Contexto: {s['context_len']}")
    
    def __repr__(self):
        return f"DatasetCollection(train={self.train}, val={self.validation}, test={self.test})"


if __name__ == "__main__":
    print("🧪 PROBANDO TEXT DATASET")
    print("="*50)
    
    # Crear tokenizador con datos
    from language.tokenizer import Tokenizer
    
    # Cargar textos
    all_texts = []
    for split in ['train', 'validation', 'test']:
        path = f"datasets/{split}.txt"
        with open(path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
    
    # Crear vocabulario
    tokenizer = Tokenizer()
    tokenizer.build_vocab(all_texts, vocab_size=200)
    
    print(f"\n📚 Vocabulario: {tokenizer.vocab_size} tokens")
    
    # Cargar cada dataset
    for split in ['train', 'validation', 'test']:
        ds = TextDataset(f"datasets/{split}.txt", tokenizer, context_len=16)
        print(f"\n{split.upper()}:")
        print(f"   Tokens: {ds.stats()['num_tokens']:,}")
        print(f"   Muestras: {ds.stats()['num_samples']:,}")
    
    # Probar batch
    ds_train = TextDataset("datasets/train.txt", tokenizer, context_len=16)
    X, y = ds_train.get_batch(batch_size=2)
    print(f"\n📦 Batch de prueba:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X[0]: {X[0][:10]}...")
    print(f"   y[0]: {y[0][:10]}...")
    
    # Collection
    print("\n")
    collection = DatasetCollection(datasets_dir="datasets", tokenizer=tokenizer, context_len=16)
    collection.summary()
    
    print("\n✅ DATASET FUNCIONANDO")
