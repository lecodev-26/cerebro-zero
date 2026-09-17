"""
Generador de embeddings sin dependencias externas.
Usa hashing + bag-of-words para convertir texto en vectores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hashlib
import re


class TextEmbedder:
    """
    Convierte texto en embeddings vectoriales.
    
    Método: hashing trick + bag-of-words + normalización L2.
    
    Uso:
        embedder = TextEmbedder(dim=128)
        vec = embedder.embed("hola mundo")
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
    
    def _normalize_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        text = text.lower()
        # Quitar puntuación
        text = re.sub(r'[^\wáéíóúñü\s]', ' ', text)
        # Colapsar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _tokenize(self, text: str) -> list:
        """Tokeniza en palabras + bigramas"""
        text = self._normalize_text(text)
        palabras = text.split()
        
        tokens = list(palabras)
        # Añadir bigramas (mejora el significado)
        for i in range(len(palabras) - 1):
            tokens.append(f"{palabras[i]}_{palabras[i+1]}")
        
        return tokens
    
    def _hash_token(self, token: str, salt: int = 0) -> int:
        """Hash estable de un token a un índice"""
        h = hashlib.md5(f"{salt}:{token}".encode('utf-8')).hexdigest()
        return int(h, 16) % self.dim
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convierte texto en un vector de dimensión `dim`.
        """
        vec = np.zeros(self.dim, dtype=np.float64)
        
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        
        # Sumar contribuciones de cada token
        for token in tokens:
            idx = self._hash_token(token)
            # También usar un segundo hash para signo (evita colisiones)
            sign = 1 if self._hash_token(token, salt=1) % 2 == 0 else -1
            vec[idx] += sign
        
        # Normalizar L2 (unitario)
        norma = np.linalg.norm(vec)
        if norma > 0:
            vec = vec / norma
        
        return vec
    
    def __repr__(self):
        return f"TextEmbedder(dim={self.dim})"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similitud coseno entre dos vectores.
    Los vectores ya están normalizados, así que es solo el producto punto.
    """
    return float(np.dot(a, b))


if __name__ == "__main__":
    print("🧪 PROBANDO TEXT EMBEDDER")
    print("="*50)
    
    embedder = TextEmbedder(dim=128)
    
    # Textos de prueba
    textos = [
        "hola mundo",
        "hola mundo cruel",
        "adiós mundo",
        "inteligencia artificial",
        "cerebro artificial",
        "python y numpy",
    ]
    
    # Vectorizar
    vectores = {t: embedder.embed(t) for t in textos}
    
    print(f"\n📊 Vectores generados (dim={embedder.dim}):")
    for t, v in vectores.items():
        print(f"   '{t}' → norma={np.linalg.norm(v):.4f}")
    
    # Similitudes
    print("\n📐 Similitudes coseno:")
    pares = [
        ("hola mundo", "hola mundo cruel"),
        ("hola mundo", "adiós mundo"),
        ("inteligencia artificial", "cerebro artificial"),
        ("inteligencia artificial", "python y numpy"),
    ]
    for a, b in pares:
        sim = cosine_similarity(vectores[a], vectores[b])
        print(f"   '{a}' vs '{b}': {sim:.4f}")
    
    print("\n✅ TEXT EMBEDDER FUNCIONANDO")
