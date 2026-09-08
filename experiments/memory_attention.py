"""
Integración de memoria y atención
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.retrieval import MemoryRetrieval
from core.transformer import Transformer
from language.tokenizer import Tokenizer

class MemoryAttention:
    def __init__(self):
        self.memory = MemoryRetrieval()
        self.tokenizer = Tokenizer()
        
        try:
            self.tokenizer.load("vocab.json")
            print("📚 Vocabulario cargado")
        except:
            print("⚠️ No se encontró vocab.json")
            print("   Usando tokenizador básico")
        
        self.transformer = Transformer(
            vocab_size=self.tokenizer.vocab_size or 50,
            d_model=16,
            num_heads=2,
            d_ff=32,
            num_layers=2,
            max_len=20
        )
        print("🧠 Transformer creado")
    
    def process(self, input_text):
        """
        Procesa una entrada usando memoria + atención
        """
        # 1. Recuperar memoria relevante
        memoria = self.memory.remember(input_text)
        
        # 2. Construir contexto con la memoria
        contexto = self._build_context(input_text, memoria)
        
        # 3. Codificar entrada
        if self.tokenizer.vocab_size > 0:
            tokens = self.tokenizer.encode_with_special(input_text)
        else:
            # Tokenizador básico si no hay vocabulario
            tokens = [ord(c) % 50 for c in input_text[:10]]
        
        # 4. Pasar por el Transformer
        x = np.array([tokens[:8]])
        probs = self.transformer.forward(x)
        
        # 5. Obtener predicción
        pred_idx = np.argmax(probs[0, -1])
        
        if self.tokenizer.vocab_size > 0:
            pred_text = self.tokenizer.decode([pred_idx])
        else:
            pred_text = str(pred_idx)
        
        return {
            'input': input_text,
            'memoria': memoria,
            'contexto': contexto,
            'prediccion': pred_text,
            'probabilidad': float(np.max(probs[0, -1]))
        }
    
    def _build_context(self, input_text, memoria):
        """
        Construye contexto a partir de la memoria recuperada
        """
        contexto = []
        
        # Agregar memoria a largo plazo
        for item in memoria.get('long_term', []):
            contexto.append(f"Conocimiento: {item['key']} = {item['value']}")
        
        # Agregar memoria episódica
        for item in memoria.get('episodic', []):
            contexto.append(f"Experiencia: {item['input']} → {item['output']}")
        
        return contexto
    
    def learn(self, input_text, output_text, context=None):
        """
        Aprende de una experiencia
        """
        self.memory.add_experience(input_text, output_text, context)
        print(f"🧠 Aprendido: {input_text} → {output_text}")
    
    def remember(self, query):
        """
        Recuerda información relacionada
        """
        return self.memory.remember(query)
    
    def __repr__(self):
        return f"MemoryAttention(memory={self.memory})"

def prueba_memory_attention():
    print("🧠 PROBANDO MEMORIA + ATENCIÓN")
    print("="*40)
    
    # Crear sistema
    ma = MemoryAttention()
    
    # Aprender experiencias
    ma.learn("hola", "Hola, soy Cerebro Zero", "saludo")
    ma.learn("¿cómo estás?", "Estoy bien, gracias", "saludo")
    ma.learn("qué es la ia", "La inteligencia artificial es un campo de la computación", "educacion")
    
    # Procesar consultas
    print("\n📝 Procesando consultas:")
    
    consultas = ["hola", "qué es la ia", "adiós"]
    
    for consulta in consultas:
        resultado = ma.process(consulta)
        print(f"\n   >>> Entrada: {consulta}")
        print(f"   <<< Predicción: {resultado['prediccion']}")
        print(f"   📊 Contexto: {resultado['contexto'][:2]}")
        print(f"   🎯 Probabilidad: {resultado['probabilidad']:.4f}")
    
    print("\n✅ MEMORIA + ATENCIÓN FUNCIONANDO!")

if __name__ == "__main__":
    prueba_memory_attention()
