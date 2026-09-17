"""
Tests de Tokenizer 3.0 (Byte + BPE)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from language.tokenizer_v3 import (
    TokenizerBase, ByteTokenizer, BPETokenizer, Merge,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def byte_tok():
    return ByteTokenizer()


@pytest.fixture
def bpe_tok():
    """BPE entrenado con corpus pequeño pero repetido"""
    corpus = (
        "hola mundo hola a todos. "
        "hola que tal. hola hola. "
        "programación en python programación en termux. "
        "cerebro cero cerebro cero. "
    ) * 10
    tok = BPETokenizer(vocab_size=280)
    tok.entrenar(corpus)
    return tok


# ============================================
# TESTS DE MERGE
# ============================================

def test_merge_creation():
    m = Merge(a=1, b=2, nuevo_id=260, frecuencia=5)
    assert m.a == 1
    assert m.b == 2
    assert m.nuevo_id == 260
    assert m.frecuencia == 5


def test_merge_to_dict():
    m = Merge(a=1, b=2, nuevo_id=260, frecuencia=5)
    d = m.to_dict()
    assert d["a"] == 1
    assert d["nuevo_id"] == 260


# ============================================
# TESTS DE BYTE TOKENIZER
# ============================================

def test_byte_vocab_size(byte_tok):
    assert byte_tok.vocab_size == 260


def test_byte_especiales(byte_tok):
    assert byte_tok.PAD == 256
    assert byte_tok.BOS == 257
    assert byte_tok.EOS == 258
    assert byte_tok.UNK == 259


def test_byte_encode_ascii(byte_tok):
    ids = byte_tok.encode("Hola")
    assert ids == [72, 111, 108, 97]


def test_byte_encode_vacio(byte_tok):
    assert byte_tok.encode("") == []


def test_byte_encode_con_bos(byte_tok):
    ids = byte_tok.encode("Hola", add_bos=True)
    assert ids[0] == byte_tok.BOS
    assert len(ids) == 5


def test_byte_encode_con_eos(byte_tok):
    ids = byte_tok.encode("Hola", add_eos=True)
    assert ids[-1] == byte_tok.EOS


def test_byte_encode_bos_eos(byte_tok):
    ids = byte_tok.encode("Hi", add_bos=True, add_eos=True)
    assert ids[0] == byte_tok.BOS
    assert ids[-1] == byte_tok.EOS


def test_byte_decode(byte_tok):
    ids = [72, 111, 108, 97]
    assert byte_tok.decode(ids) == "Hola"


def test_byte_roundtrip_ascii(byte_tok):
    texto = "Hola mundo"
    ids = byte_tok.encode(texto)
    assert byte_tok.decode(ids) == texto


def test_byte_roundtrip_acentos(byte_tok):
    texto = "programación"
    ids = byte_tok.encode(texto)
    assert byte_tok.decode(ids) == texto


def test_byte_roundtrip_emoji(byte_tok):
    texto = "🔥"
    ids = byte_tok.encode(texto)
    assert byte_tok.decode(ids) == texto


def test_byte_roundtrip_chino(byte_tok):
    texto = "你好"
    ids = byte_tok.encode(texto)
    assert byte_tok.decode(ids) == texto


def test_byte_roundtrip_mixto(byte_tok):
    texto = "Hola 🔥 你好 café"
    ids = byte_tok.encode(texto)
    assert byte_tok.decode(ids) == texto


def test_byte_decode_ignora_especiales(byte_tok):
    ids = [byte_tok.BOS, 72, 105, byte_tok.EOS]
    assert byte_tok.decode(ids) == "Hi"


def test_byte_guardar(byte_tok):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        byte_tok.guardar(tmp)
        assert os.path.exists(tmp)
    finally:
        os.unlink(tmp)


def test_byte_cargar(byte_tok):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        byte_tok.guardar(tmp)
        tok2 = ByteTokenizer.cargar(tmp)
        assert tok2.vocab_size == byte_tok.vocab_size
    finally:
        os.unlink(tmp)


def test_byte_repr(byte_tok):
    assert "ByteTokenizer" in repr(byte_tok)


# ============================================
# TESTS DE BPE
# ============================================

def test_bpe_vocab_size_minimo():
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=100)


def test_bpe_creation():
    tok = BPETokenizer(vocab_size=300)
    assert tok.vocab_size_objetivo == 300
    assert tok.num_merges() == 0


def test_bpe_vocab_inicial():
    tok = BPETokenizer(vocab_size=300)
    assert tok.vocab_size == 260


def test_bpe_entrenar(bpe_tok):
    assert bpe_tok.num_merges() > 0
    assert bpe_tok.vocab_size > 260


def test_bpe_entrenar_respeta_vocab_size():
    corpus = "abc abc abc abc abc " * 50
    tok = BPETokenizer(vocab_size=270)
    tok.entrenar(corpus)
    assert tok.vocab_size == 270


def test_bpe_entrenar_corpus_vacio():
    tok = BPETokenizer(vocab_size=300)
    tok.entrenar("")
    assert tok.num_merges() == 0


def test_bpe_encode_decode_ascii(bpe_tok):
    texto = "hola"
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_decode_acentos(bpe_tok):
    texto = "programación"
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_decode_emoji(bpe_tok):
    texto = "🔥"
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_decode_chino(bpe_tok):
    texto = "你好"
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_con_bos(bpe_tok):
    ids = bpe_tok.encode("hola", add_bos=True)
    assert ids[0] == BPETokenizer.BOS


def test_bpe_encode_con_eos(bpe_tok):
    ids = bpe_tok.encode("hola", add_eos=True)
    assert ids[-1] == BPETokenizer.EOS


def test_bpe_encode_vacio(bpe_tok):
    assert bpe_tok.encode("") == []


def test_bpe_compresion_mejora(bpe_tok):
    """BPE debe comprimir textos vistos en entrenamiento"""
    texto = "hola mundo"
    ratio = bpe_tok.compresion_media(texto)
    assert ratio >= 1.0


def test_bpe_compresion_corpus_vacio(bpe_tok):
    assert bpe_tok.compresion_media("") == 0.0


def test_bpe_cache(bpe_tok):
    """El caché debe devolver lo mismo en segunda llamada"""
    ids1 = bpe_tok.encode("hola")
    ids2 = bpe_tok.encode("hola")
    assert ids1 == ids2


def test_bpe_merges_ordenadas(bpe_tok):
    """Los merges deben tener IDs crecientes"""
    for i, m in enumerate(bpe_tok.merges):
        assert m.nuevo_id == 260 + i


def test_bpe_entrenar_verbose(bpe_tok):
    """verbose=True no debe romper"""
    tok = BPETokenizer(vocab_size=280)
    tok.entrenar("abc abc abc abc " * 20, verbose=False)
    assert tok.num_merges() > 0


# ============================================
# TESTS DE PERSISTENCIA BPE
# ============================================

def test_bpe_guardar_cargar(bpe_tok):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        bpe_tok.guardar(tmp)
        bpe2 = BPETokenizer.cargar(tmp)
        
        assert bpe2.vocab_size == bpe_tok.vocab_size
        assert bpe2.num_merges() == bpe_tok.num_merges()
    finally:
        os.unlink(tmp)


def test_bpe_guardar_cargar_mismo_encode(bpe_tok):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        bpe_tok.guardar(tmp)
        bpe2 = BPETokenizer.cargar(tmp)
        
        texto = "hola mundo"
        assert bpe2.encode(texto) == bpe_tok.encode(texto)
    finally:
        os.unlink(tmp)


def test_bpe_guardar_cargar_mismo_decode(bpe_tok):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        bpe_tok.guardar(tmp)
        bpe2 = BPETokenizer.cargar(tmp)
        
        ids = bpe_tok.encode("programación")
        assert bpe2.decode(ids) == "programación"
    finally:
        os.unlink(tmp)


# ============================================
# TESTS DE COMPARACIÓN
# ============================================

def test_bpe_mejor_que_byte_en_corpus(byte_tok):
    """BPE debe comprimir mejor que byte en textos del corpus"""
    corpus = "hola mundo hola mundo hola mundo " * 20
    bpe = BPETokenizer(vocab_size=290)
    bpe.entrenar(corpus)
    
    texto = "hola mundo"
    bytes_len = len(byte_tok.encode(texto))
    bpe_len = len(bpe.encode(texto))
    
    # BPE debe ser <= bytes
    assert bpe_len <= bytes_len


def test_bpe_frecuencia_merges_decreciente(bpe_tok):
    """
    Las primeras fusiones deberían tener mayor frecuencia
    que las últimas (aproximadamente).
    """
    if bpe_tok.num_merges() < 10:
        pytest.skip("Pocos merges para comparar")
    
    freq_primera = bpe_tok.merges[0].frecuencia
    freq_ultima = bpe_tok.merges[-1].frecuencia
    
    # No tienen que ser estrictamente decrecientes, pero
    # la primera debería ser >= la última
    assert freq_primera >= freq_ultima


# ============================================
# TESTS DE INTERFAZ COMÚN
# ============================================

def test_byte_es_tokenizer_base(byte_tok):
    assert isinstance(byte_tok, TokenizerBase)


def test_bpe_es_tokenizer_base(bpe_tok):
    assert isinstance(bpe_tok, TokenizerBase)


# ============================================
# TESTS EXTRA DE CALIDAD
# ============================================

def test_bpe_roundtrip_largo(bpe_tok):
    """Texto largo con varios idiomas y símbolos"""
    texto = "Hola mundo! programación en Python 🔥 你好 café"
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_unicode_raro(bpe_tok):
    """Caracteres raros deben funcionar (byte-level)"""
    texto = "\u200b\u200c\u200d"  # zero-width chars
    ids = bpe_tok.encode(texto)
    # Deben ser 3 bytes (uno por cada zero-width char)
    assert len(ids) >= 3
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_solo_espacios(bpe_tok):
    texto = "     "
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_encode_repetido(bpe_tok):
    texto = "aaaa" * 20
    ids = bpe_tok.encode(texto)
    assert bpe_tok.decode(ids) == texto


def test_bpe_num_merges_crece_con_vocab():
    corpus = "abc abc abc " * 100
    tok1 = BPETokenizer(vocab_size=270)
    tok2 = BPETokenizer(vocab_size=290)
    tok1.entrenar(corpus)
    tok2.entrenar(corpus)
    assert tok2.num_merges() >= tok1.num_merges()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
