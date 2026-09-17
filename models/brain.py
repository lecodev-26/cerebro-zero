"""
Interfaz base para todos los modelos de Cerebro Zero
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Brain(ABC):
    """
    Interfaz base para todos los cerebros.
    
    Todas las implementaciones (MLP, CNN, Transformer, ...) deben heredar
    de esta clase y sobrescribir sus métodos.
    """
    
    def __init__(self, name: str = "Brain"):
        self.name = name
        self.version = "1.0.0"
        self._trained = False
        self._training_history = []
    
    # ============================================
    # MÉTODOS ABSTRACTOS (a implementar por subclases)
    # ============================================
    
    @abstractmethod
    def forward(self, x):
        """
        Computación hacia adelante.
        x: entrada (Tensor)
        return: salida (Tensor)
        """
        pass
    
    @abstractmethod
    def parameters(self):
        """
        Devuelve los parámetros entrenables del modelo.
        return: lista de Tensores
        """
        pass
    
    # ============================================
    # MÉTODOS CON IMPLEMENTACIÓN POR DEFECTO
    # ============================================
    
    def predict(self, x):
        """
        Predicción (sin gradientes).
        """
        return self.forward(x)
    
    def train(self, dataset, epochs: int = 100, lr: float = 0.01, 
              optimizer=None, loss_fn=None, verbose: bool = True):
        """
        Entrenamiento genérico.
        
        dataset: iterable de (X, y)
        epochs: número de épocas
        lr: learning rate
        optimizer: optimizador (si None, se crea uno por defecto)
        loss_fn: función de pérdida (si None, se crea MSE)
        verbose: mostrar progreso
        """
        from core.optimizers import Adam
        from core.losses import MSELoss
        
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr=lr)
        if loss_fn is None:
            loss_fn = MSELoss()
        
        if verbose:
            print(f"🏋️ Entrenando {self.name} ({epochs} épocas)...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for X, y in dataset:
                pred = self.forward(X)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.data
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            self._training_history.append(avg_loss)
            
            if verbose and epoch % max(epochs // 10, 1) == 0:
                print(f"   Epoch {epoch}: loss = {avg_loss:.6f}")
        
        self._trained = True
        if verbose:
            print(f"✅ Entrenamiento completado")
        
        return self._training_history
    
    def save(self, path: str):
        """
        Guarda el modelo en disco (pickle).
        """
        import pickle
        state = self._get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 {self.name} guardado en {path}")
    
    def load(self, path: str):
        """
        Carga el modelo desde disco.
        """
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._set_state(state)
        self._trained = True
        print(f"📂 {self.name} cargado desde {path}")
    
    def _get_state(self) -> Dict[str, Any]:
        """
        Devuelve el estado del modelo (a sobrescribir si es necesario).
        Por defecto guarda los parámetros.
        """
        return {
            'name': self.name,
            'version': self.version,
            'trained': self._trained,
            'history': self._training_history,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        """
        Restaura el estado del modelo.
        """
        self.name = state.get('name', self.name)
        self.version = state.get('version', self.version)
        self._trained = state.get('trained', False)
        self._training_history = state.get('history', [])
    
    # ============================================
    # UTILIDADES
    # ============================================
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_history(self):
        return self._training_history
    
    def num_parameters(self) -> int:
        """Cuenta el número total de parámetros"""
        total = 0
        for p in self.parameters():
            if hasattr(p, 'data'):
                total += p.data.size
        return total
    
    def summary(self):
        """Muestra un resumen del modelo"""
        print(f"🧠 {self.name} v{self.version}")
        print(f"   Entrenado: {'✅' if self._trained else '❌'}")
        print(f"   Parámetros: {self.num_parameters():,}")
        if self._training_history:
            print(f"   Épocas: {len(self._training_history)}")
            print(f"   Loss final: {self._training_history[-1]:.6f}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
