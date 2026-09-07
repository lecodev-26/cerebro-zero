import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import json
from datetime import datetime

class MemoriaPersistente:
    def __init__(self, archivo="memoria.json", max_size=1000):
        self.archivo = archivo
        self.max_size = max_size
        self.experiencias = []
        self.embeddings = []
        self.cargar()
    
    def guardar(self):
        data = {
            'experiencias': self.experiencias,
            'embeddings': self.embeddings,
            'fecha_actualizacion': datetime.now().isoformat()
        }
        with open(os.path.join("../modelos_guardados", self.archivo), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Memoria guardada en modelos_guardados/{self.archivo}")
    
    def cargar(self):
        try:
            path = os.path.join("../modelos_guardados", self.archivo)
            with open(path, 'r') as f:
                data = json.load(f)
            self.experiencias = data.get('experiencias', [])
            self.embeddings = data.get('embeddings', [])
            print(f"📂 Memoria cargada: {len(self.experiencias)} experiencias")
        except:
            print("📂 Memoria nueva (vacía)")
            self.experiencias = []
            self.embeddings = []
    
    def recordar(self, entrada, k=3):
        if not self.experiencias:
            return []
        
        entrada = np.array(entrada).flatten()
        if len(self.embeddings) != len(self.experiencias):
            self.embeddings = self._embed_experiencias()
        
        # Calcular similitud
        similitudes = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(entrada, emb) / (np.linalg.norm(entrada) * np.linalg.norm(emb) + 1e-10)
            similitudes.append((i, sim))
        
        # Ordenar por similitud (de mayor a menor)
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver las k más similares (solo si similitud > 0.3)
        resultados = []
        for i, sim in similitudes[:k]:
            if sim > 0.3:
                resultados.append(self.experiencias[i])
        
        return resultados
    
    def aprender(self, entrada, salida):
        if len(self.experiencias) >= self.max_size:
            self.experiencias.pop(0)
            self.embeddings.pop(0)
        
        experiencia = {
            'entrada': entrada.tolist() if isinstance(entrada, np.ndarray) else entrada,
            'salida': salida.tolist() if isinstance(salida, np.ndarray) else salida,
            'fecha': datetime.now().isoformat()
        }
        self.experiencias.append(experiencia)
        
        emb = np.array(entrada).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        self.embeddings.append(emb.tolist())
        
        self.guardar()
    
    def olvidar(self, indice=None):
        if indice is None:
            self.experiencias = []
            self.embeddings = []
            print("🧠 Memoria completamente olvidada")
        else:
            if 0 <= indice < len(self.experiencias):
                del self.experiencias[indice]
                del self.embeddings[indice]
                print(f"🧠 Experiencia {indice} olvidada")
        self.guardar()
    
    def resumen(self):
        print(f"🧠 MEMORIA PERSISTENTE")
        print(f"   Experiencias: {len(self.experiencias)}")
        print(f"   Capacidad máxima: {self.max_size}")
        print(f"   Última actualización: {self.experiencias[-1].get('fecha', 'N/A') if self.experiencias else 'N/A'}")
        if self.experiencias:
            print(f"   Ejemplo de entrada: {self.experiencias[0]['entrada']}")
    
    def _embed_experiencias(self):
        embeddings = []
        for exp in self.experiencias:
            emb = np.array(exp['entrada']).flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            embeddings.append(emb.tolist())
        return embeddings
