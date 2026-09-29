class TokenizerContract:
    version="1"
    def encode(self,text): return text.split()
    def decode(self,tokens): return " ".join(tokens)
