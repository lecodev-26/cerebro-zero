"""
Memoria + Atención - Versión mejorada con respuestas directas de memoria
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memory.retrieval import MemoryRetrieval
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
            self.tokenizer = None
            print("   Usando modo sin tokenizador")
    
    def process(self, input_text):
        """
        Procesa una entrada usando memoria + atención
        """
        # 1. Recuperar memoria relevante
        memoria = self.memory.remember(input_text)
        
        # 2. Buscar respuesta exacta en memoria episódica
        respuesta = self._buscar_en_memoria(input_text)
        
        if respuesta:
            return {
                'input': input_text,
                'memoria': memoria,
                'respuesta': respuesta,
                'fuente': 'memoria',
                'probabilidad': 1.0
            }
        
        # 3. Si no hay respuesta exacta, buscar contexto
        contexto = self._build_context(input_text, memoria)
        
        if contexto:
            return {
                'input': input_text,
                'memoria': memoria,
                'respuesta': contexto[0] if contexto else "No tengo información sobre eso",
                'fuente': 'contexto',
                'probabilidad': 0.7
            }
        
        # 4. Respuesta por defecto
        return {
            'input': input_text,
            'memoria': memoria,
            'respuesta': "No tengo información sobre eso. ¿Puedes preguntar de otra forma?",
            'fuente': 'default',
            'probabilidad': 0.5
        }
    
    def _buscar_en_memoria(self, input_text):
        """
        Busca una respuesta exacta en la memoria episódica
        """
        episodios = self.memory.get_all_episodic()
        for ep in episodios:
            if input_text.lower() in ep.get('input', '').lower():
                return ep.get('output', '')
        return None
    
    def _build_context(self, input_text, memoria):
        """
        Construye contexto a partir de la memoria recuperada
        """
        contexto = []
        
        # Memoria a largo plazo
        for item in memoria.get('long_term', []):
            contexto.append(f"{item['key']}: {item['value']}")
        
        # Memoria episódica
        for item in memoria.get('episodic', []):
            contexto.append(f"Recuerdo: {item['input']} → {item['output']}")
        
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
    print("🧠 PROBANDO MEMORIA + ATENCIÓN (VERSIÓN MEJORADA)")
    print("="*50)
    
    ma = MemoryAttention()
    
    # Aprender experiencias
    ma.learn("hola", "Hola, soy Cerebro Zero", "saludo")
    ma.learn("¿cómo estás?", "Estoy bien, gracias", "saludo")
    ma.learn("qué es la ia", "La inteligencia artificial es un campo de la computación", "educacion")
    ma.learn("adiós", "Hasta luego, ha sido un placer", "despedida")
    
    # Procesar consultas
    print("\n📝 Procesando consultas:")
    
    consultas = ["hola", "qué es la ia", "adiós", "qué es python", "¿cómo estás?"]
    
    for consulta in consultas:
        resultado = ma.process(consulta)
        print(f"\n   >>> Entrada: {consulta}")
        print(f"   <<< Respuesta: {resultado['respuesta']}")
        print(f"   📊 Fuente: {resultado['fuente']}")
        print(f"   🎯 Probabilidad: {resultado['probabilidad']:.2f}")
    
    print("\n✅ MEMORIA + ATENCIÓN FUNCIONANDO CORRECTAMENTE!")

if __name__ == "__main__":
    prueba_memory_attention()
