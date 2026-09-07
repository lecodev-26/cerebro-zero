import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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
        
        # Convertir entrada a array numérico
        try:
            entrada = np.array(entrada).flatten().astype(float)
        except:
            # Si no se puede convertir a números, devolver experiencias recientes
            return self.experiencias[-k:] if self.experiencias else []
        
        if len(self.embeddings) != len(self.experiencias):
            self.embeddings = self._embed_experiencias()
        
        similitudes = []
        for i, emb in enumerate(self.embeddings):
            try:
                sim = np.dot(entrada, emb) / (np.linalg.norm(entrada) * np.linalg.norm(emb) + 1e-10)
                similitudes.append((i, sim))
            except:
                similitudes.append((i, 0))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        resultados = []
        for i, sim in similitudes[:k]:
            if sim > 0.1:  # Umbral más bajo
                resultados.append(self.experiencias[i])
        
        return resultados if resultados else self.experiencias[-k:]
    
    def aprender(self, entrada, salida):
        # Asegurar que entrada es una lista de números
        try:
            entrada_lista = entrada.tolist() if isinstance(entrada, np.ndarray) else entrada
            salida_lista = salida.tolist() if isinstance(salida, np.ndarray) else salida
        except:
            entrada_lista = [float(x) if isinstance(x, (int, float)) else 0 for x in entrada]
            salida_lista = [float(x) if isinstance(x, (int, float)) else 0 for x in salida]
        
        if len(self.experiencias) >= self.max_size:
            self.experiencias.pop(0)
            self.embeddings.pop(0)
        
        experiencia = {
            'entrada': entrada_lista,
            'salida': salida_lista,
            'fecha': datetime.now().isoformat()
        }
        self.experiencias.append(experiencia)
        
        # Embedding para búsqueda
        try:
            emb = np.array(entrada_lista).flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            self.embeddings.append(emb.tolist())
        except:
            self.embeddings.append([0.0] * len(entrada_lista))
        
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
        if self.experiencias:
            print(f"   Última actualización: {self.experiencias[-1].get('fecha', 'N/A')}")
            print(f"   Ejemplo de entrada: {self.experiencias[0].get('entrada', 'N/A')}")
    
    def _embed_experiencias(self):
        embeddings = []
        for exp in self.experiencias:
            try:
                emb = np.array(exp.get('entrada', [0])).flatten()
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                embeddings.append(emb.tolist())
            except:
                embeddings.append([0.0])
        return embeddings
