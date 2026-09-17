"""
Tokenizer 3.0 — Byte-level + BPE desde cero
=============================================

Filosofía:
    Sin vocabulario predefinido. Sin diccionarios externos.
    Bytes puros + fusiones aprendidas del corpus.

Componentes:
- TokenizerBase: interfaz común
- ByteTokenizer: cada byte → un token (256 posibles)
- BPETokenizer: Byte Pair Encoding entrenado desde cero

Ventajas:
- Cualquier texto (emoji, acentos, chino, símbolos raros)
- Sin tokens <UNK>
- Vocabulario ajustable (tamaño objetivo)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter


# ============================================
# BASE
# ============================================

class TokenizerBase:
    """Interfaz común de tokenizers"""
    
    def encode(self, texto: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
    
    def guardar(self, archivo: str):
        raise NotImplementedError
    
    @classmethod
    def cargar(cls, archivo: str) -> "TokenizerBase":
        raise NotImplementedError


# ============================================
# BYTE TOKENIZER
# ============================================

class ByteTokenizer(TokenizerBase):
    """
    Tokenizer byte-level puro.
    
    Vocabulario:
    - 0..255: bytes
    - 256: PAD
    - 257: BOS
    - 258: EOS
    - 259: UNK (reservado por compatibilidad)
    
    Total: 260 tokens
    """
    
    PAD = 256
    BOS = 257
    EOS = 258
    UNK = 259
    
    def __init__(self):
        self.especiales = {
            "PAD": self.PAD,
            "BOS": self.BOS,
            "EOS": self.EOS,
            "UNK": self.UNK,
        }
    
    @property
    def vocab_size(self) -> int:
        return 260
    
    def encode(self, texto: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Texto → lista de ids de byte.
        """
        if not isinstance(texto, str):
            texto = str(texto)
        
        ids = []
        if add_bos:
            ids.append(self.BOS)
        
        for b in texto.encode("utf-8"):
            ids.append(b)
        
        if add_eos:
            ids.append(self.EOS)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Lista de ids → texto.
        Ignora tokens especiales.
        """
        bytes_list = []
        for i in ids:
            if i < 256:
                bytes_list.append(i)
        try:
            return bytes(bytes_list).decode("utf-8", errors="replace")
        except Exception:
            return ""
    
    def guardar(self, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        data = {"tipo": "byte", "especiales": self.especiales}
        with open(archivo, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def cargar(cls, archivo: str) -> "ByteTokenizer":
        return cls()
    
    def __repr__(self):
        return f"ByteTokenizer(vocab={self.vocab_size})"


# ============================================
# BPE TOKENIZER
# ============================================

@dataclass
class Merge:
    """Una fusión BPE: dos tokens se convierten en uno nuevo"""
    a: int
    b: int
    nuevo_id: int
    frecuencia: int = 0
    
    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "nuevo_id": self.nuevo_id,
                "frecuencia": self.frecuencia}


class BPETokenizer(TokenizerBase):
    """
    Byte Pair Encoding implementado desde cero.
    
    Algoritmo:
    1. Empezar con bytes (256 tokens)
    2. Contar pares de tokens adyacentes en el corpus
    3. Fusionar el par más frecuente → nuevo token
    4. Repetir hasta alcanzar vocab_size objetivo
    
    Encoding:
    - Aplicar fusiones en orden de creación (las aprendidas primero, primero)
    - Resultado: secuencia corta de ids
    """
    
    PAD = 256
    BOS = 257
    EOS = 258
    
    def __init__(self, vocab_size: int = 300):
        if vocab_size < 260:
            raise ValueError("vocab_size debe ser >= 260 (256 bytes + 4 especiales)")
        
        self.vocab_size_objetivo = vocab_size
        self.merges: List[Merge] = []
        self._caches_encode: Dict[str, List[int]] = {}
    
    # ============================================
    # ENTRENAMIENTO
    # ============================================
    
    def entrenar(self, corpus: str, verbose: bool = False):
        """
        Aprende fusiones BPE del corpus.
        
        Args:
            corpus: texto de entrenamiento
            verbose: mostrar progreso
        """
        if not isinstance(corpus, str):
            corpus = str(corpus)
        
        # 1. Empezar con bytes
        tokens: List[int] = list(corpus.encode("utf-8"))
        
        # 2. Fusionar iterativamente
        num_fusiones = self.vocab_size_objetivo - 260
        
        for i in range(num_fusiones):
            # Contar pares
            pares = Counter()
            for j in range(len(tokens) - 1):
                pares[(tokens[j], tokens[j + 1])] += 1
            
            if not pares:
                break
            
            # Par más frecuente
            (a, b), freq = pares.most_common(1)[0]
            
            if freq < 2:
                # Ya no hay pares útiles
                break
            
            nuevo_id = 260 + len(self.merges)
            
            # Registrar merge
            self.merges.append(Merge(a=a, b=b, nuevo_id=nuevo_id, frecuencia=freq))
            
            # Aplicar fusión al corpus tokenizado
            nuevos_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == a and tokens[j + 1] == b:
                    nuevos_tokens.append(nuevo_id)
                    j += 2
                else:
                    nuevos_tokens.append(tokens[j])
                    j += 1
            tokens = nuevos_tokens
            
            if verbose and (i + 1) % 20 == 0:
                print(f"   Merge {i+1}/{num_fusiones}: ({a},{b})→{nuevo_id} freq={freq}")
        
        # Invalidar caché
        self._caches_encode.clear()
    
    # ============================================
    # ENCODING
    # ============================================
    
    def encode(self, texto: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Aplica fusiones en orden para obtener tokens finales.
        """
        if not isinstance(texto, str):
            texto = str(texto)
        
        # Caché
        cache_key = texto
        if cache_key in self._caches_encode:
            ids = self._caches_encode[cache_key].copy()
        else:
            ids = list(texto.encode("utf-8"))
            
            # Aplicar cada merge en orden
            for merge in self.merges:
                nuevos = []
                j = 0
                while j < len(ids):
                    if (j < len(ids) - 1
                            and ids[j] == merge.a
                            and ids[j + 1] == merge.b):
                        nuevos.append(merge.nuevo_id)
                        j += 2
                    else:
                        nuevos.append(ids[j])
                        j += 1
                ids = nuevos
            
            self._caches_encode[cache_key] = ids.copy()
        
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        ids → texto.
        
        1. Expandir cada id a sus bytes (recursivamente si es merge)
        2. Decodificar utf-8
        """
        # Mapa id → bytes
        bytes_map: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for merge in self.merges:
            if merge.a in bytes_map and merge.b in bytes_map:
                bytes_map[merge.nuevo_id] = bytes_map[merge.a] + bytes_map[merge.b]
        
        # Expandir
        out = bytearray()
        for i in ids:
            if i in bytes_map:
                out.extend(bytes_map[i])
            # Especiales se ignoran
        
        try:
            return out.decode("utf-8", errors="replace")
        except Exception:
            return ""
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    @property
    def vocab_size(self) -> int:
        return 260 + len(self.merges)
    
    def num_merges(self) -> int:
        return len(self.merges)
    
    def compresion_media(self, texto: str) -> float:
        """
        Ratio de compresión: bytes_original / tokens_BPE.
        Cuanto mayor, más comprime.
        """
        bytes_len = len(texto.encode("utf-8"))
        tokens_len = len(self.encode(texto))
        if tokens_len == 0:
            return 0.0
        return bytes_len / tokens_len
    
    # ============================================
    # PERSISTENCIA
    # ============================================
    
    def guardar(self, archivo: str):
        os.makedirs(os.path.dirname(archivo) or ".", exist_ok=True)
        data = {
            "tipo": "bpe",
            "vocab_size_objetivo": self.vocab_size_objetivo,
            "merges": [m.to_dict() for m in self.merges],
        }
        with open(archivo, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def cargar(cls, archivo: str) -> "BPETokenizer":
        with open(archivo, "r") as f:
            data = json.load(f)
        
        tok = cls(vocab_size=data["vocab_size_objetivo"])
        tok.merges = [
            Merge(a=m["a"], b=m["b"], nuevo_id=m["nuevo_id"],
                  frecuencia=m.get("frecuencia", 0))
            for m in data["merges"]
        ]
        return tok
    
    def __repr__(self):
        return f"BPETokenizer(vocab={self.vocab_size}, merges={len(self.merges)})"


# ============================================
# TEST MANUAL
# ============================================

if __name__ == "__main__":
    print("🧪 PROBANDO TOKENIZER 3.0")
    print("=" * 60)
    
    # ============================================
    # BYTE TOKENIZER
    # ============================================
    print("\n📦 ByteTokenizer:")
    bt = ByteTokenizer()
    print(f"   {bt}")
    
    textos = ["Hola", "Hello", "Barcelona", "programación", "🔥", "你好"]
    print(f"\n   Encoding de ejemplos:")
    for t in textos:
        ids = bt.encode(t)
        dec = bt.decode(ids)
        print(f"   '{t}' → {ids} → '{dec}' {'✅' if dec == t else '❌'}")
    
    # ============================================
    # BPE TOKENIZER
    # ============================================
    print(f"\n🔤 BPETokenizer:")
    
    # Corpus pequeño
    corpus = (
        "hola mundo hola mundo hola a todos. "
        "hola que tal. hola hola. "
        "programación en python programación en termux. "
        "cerebro cero cerebro cero. "
        "🔥 emoji y texto 🔥 mezclados. "
    ) * 5  # repetir para tener frecuencia
    
    print(f"   Corpus: {len(corpus)} caracteres")
    
    bpe = BPETokenizer(vocab_size=300)
    print(f"   Entrenando...")
    bpe.entrenar(corpus, verbose=False)
    print(f"   {bpe}")
    print(f"   Fusiones aprendidas: {bpe.num_merges()}")
    
    # Mostrar algunas fusiones
    print(f"\n   Primeras 5 fusiones:")
    for m in bpe.merges[:5]:
        # Ver qué bytes representa
        try:
            a_char = bytes([m.a]).decode("utf-8", errors="replace") if m.a < 256 else f"<{m.a}>"
            b_char = bytes([m.b]).decode("utf-8", errors="replace") if m.b < 256 else f"<{m.b}>"
        except Exception:
            a_char, b_char = str(m.a), str(m.b)
        print(f"   ({m.a},{m.b})→{m.nuevo_id}  freq={m.frecuencia}  '{a_char}'+'{b_char}'")
    
    # Encoding con BPE
    print(f"\n   Encoding con BPE:")
    for t in textos:
        ids_bpe = bpe.encode(t)
        ids_byte = bt.encode(t)
        dec = bpe.decode(ids_bpe)
        ok = "✅" if dec == t else "❌"
        ratio = len(ids_byte) / max(len(ids_bpe), 1)
        print(f"   '{t}'")
        print(f"      BPE:  {ids_bpe}  ({len(ids_bpe)} tokens)")
        print(f"      Byte: {len(ids_byte)} tokens  → ratio {ratio:.2f}x")
        print(f"      Decode: '{dec}' {ok}")
    
    # Compresión media
    print(f"\n   Compresión sobre corpus:")
    print(f"      Ratio: {bpe.compresion_media(corpus):.2f}x")
    
    # Persistencia
    print(f"\n💾 Test persistencia:")
    tmp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_bpe_tmp.json"
    )
    bpe.guardar(tmp)
    bpe2 = BPETokenizer.cargar(tmp)
    print(f"   Guardado/cargado: {bpe2}")
    print(f"   Mismo vocab: {bpe.vocab_size == bpe2.vocab_size}")
    print(f"   Mismo encode: {bpe.encode('hola') == bpe2.encode('hola')}")
    os.unlink(tmp)
    
    print("\n✅ TOKENIZER 3.0 FUNCIONANDO")
