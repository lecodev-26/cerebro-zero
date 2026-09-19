"""
Prueba de lenguaje: Tokenizador + Vocabulario
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.tokenizer import Tokenizer

print("🧠 PROBANDO SISTEMA DE LENGUAJE")
print("="*40)

# Textos para entrenar el vocabulario
textos = [
    "el cerebro zero es un sistema de inteligencia artificial",
    "aprende desde cero sin frameworks",
    "usa redes neuronales y memoria",
    "también puede procesar lenguaje",
    "el objetivo es crear un agente inteligente",
    "que pueda entender y generar texto",
    "esto es solo el principio de un gran proyecto"
]

# Crear tokenizador
tokenizer = Tokenizer()
tokenizer.build_vocab(textos, vocab_size=50)

print("\n📝 Ejemplos de codificación:")
for texto in textos[:3]:
    ids = tokenizer.encode_with_special(texto)
    print(f"   '{texto[:30]}...' → {ids[:10]}...")

# Guardar vocabulario
tokenizer.save("vocab.json")
print("\n✅ SISTEMA DE LENGUAJE FUNCIONANDO!")
