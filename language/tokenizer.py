"""
Tokenizador para Cerebro Zero
Convierte texto en tokens (números)
"""

import re
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.vocab_size = 0
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of Sentence
            '<EOS>': 3,  # End of Sentence
            '<SEP>': 4   # Separator
        }
    
    def build_vocab(self, texts, vocab_size=10000):
        """
        Construye el vocabulario a partir de una lista de textos
        """
        # Contar palabras
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize_text(text)
            word_counts.update(tokens)
        
        # Seleccionar las palabras más comunes
        most_common = word_counts.most_common(vocab_size - len(self.special_tokens))
        
        # Construir vocabulario
        self.vocab = self.special_tokens.copy()
        for idx, (word, _) in enumerate(most_common, start=len(self.special_tokens)):
            self.vocab[word] = idx
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"📚 Vocabulario construido: {self.vocab_size} palabras")
        return self.vocab
    
    def tokenize_text(self, text):
        """
        Convierte texto en tokens (palabras)
        """
        # Limpiar texto
        text = text.lower()
        # Reemplazar puntuación con espacios
        text = re.sub(r'[^a-záéíóúñ\s]', ' ', text)
        # Dividir en palabras
        return text.split()
    
    def encode(self, text):
        """
        Convierte texto en IDs (números)
        """
        tokens = self.tokenize_text(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['<UNK>'])
        return ids
    
    def decode(self, ids):
        """
        Convierte IDs en texto
        """
        tokens = []
        for idx in ids:
            if idx in self.inv_vocab:
                token = self.inv_vocab[idx]
                if token not in self.special_tokens:
                    tokens.append(token)
        return ' '.join(tokens)
    
    def encode_with_special(self, text, add_bos=True, add_eos=True):
        """
        Codifica con tokens especiales
        """
        ids = []
        if add_bos:
            ids.append(self.vocab['<BOS>'])
        ids.extend(self.encode(text))
        if add_eos:
            ids.append(self.vocab['<EOS>'])
        return ids
    
    def save(self, filename="vocab.json"):
        """
        Guarda el vocabulario en un archivo
        """
        import json
        with open(filename, 'w') as f:
            json.dump(self.vocab, f)
        print(f"💾 Vocabulario guardado en {filename}")
    
    def load(self, filename="vocab.json"):
        """
        Carga el vocabulario desde un archivo
        """
        import json
        with open(filename, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"📂 Vocabulario cargado: {self.vocab_size} palabras")

def prueba_tokenizer():
    print("🧠 PROBANDO TOKENIZADOR")
    print("="*30)
    
    # Textos de ejemplo
    textos = [
        "hola mundo",
        "este es un ejemplo de tokenizador",
        "el cerebro zero aprende lenguaje",
        "inteligencia artificial desde cero"
    ]
    
    # Crear tokenizer
    tokenizer = Tokenizer()
    tokenizer.build_vocab(textos, vocab_size=20)
    
    # Probar codificación
    texto = "hola mundo desde cero"
    ids = tokenizer.encode_with_special(texto)
    texto_decodificado = tokenizer.decode(ids)
    
    print(f"📝 Texto original: {texto}")
    print(f"🔢 IDs: {ids}")
    print(f"📝 Texto decodificado: {texto_decodificado}")
    
    # Guardar
    tokenizer.save("vocab_test.json")
    print("\n✅ TOKENIZADOR FUNCIONANDO!")

if __name__ == "__main__":
    prueba_tokenizer()
