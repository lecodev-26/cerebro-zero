"""
Vocabulario para Cerebro Zero
"""

import json
from language.tokenizer import Tokenizer

class Vocabulary:
    def __init__(self, vocab_file="vocab.json"):
        self.tokenizer = Tokenizer()
        self.tokenizer.load(vocab_file)
    
    def word_to_id(self, word):
        return self.tokenizer.vocab.get(word, self.tokenizer.vocab['<UNK>'])
    
    def id_to_word(self, idx):
        return self.tokenizer.inv_vocab.get(idx, '<UNK>')
    
    def size(self):
        return self.tokenizer.vocab_size
    
    def encode(self, text):
        return self.tokenizer.encode_with_special(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids)

def prueba_vocabulario():
    print("🧠 PROBANDO VOCABULARIO")
    print("="*30)
    
    # Crear vocabulario desde un archivo existente
    try:
        vocab = Vocabulary("vocab_test.json")
        print(f"📚 Vocabulario cargado: {vocab.size()} palabras")
        
        # Probar
        texto = "hola mundo"
        ids = vocab.encode(texto)
        print(f"📝 '{texto}' → {ids}")
        print(f"📝 Decodificado: {vocab.decode(ids)}")
        
    except:
        print("❌ No se encontró vocab_test.json")
        print("   Ejecuta primero: python language/tokenizer.py")

if __name__ == "__main__":
    prueba_vocabulario()
