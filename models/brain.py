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
    """
    
    def __init__(self, name: str = "Brain"):
        self.name = name
        self.version = "1.0.0"
        self._trained = False
        self._training_history = []
    
    # ============================================
    # MÉTODOS ABSTRACTOS
    # ============================================
    
    @abstractmethod
    def forward(self, x):
        """Computación hacia adelante"""
        pass
    
    @abstractmethod
    def parameters(self):
        """Devuelve los parámetros entrenables"""
        pass
    
    # ============================================
    # MÉTODOS CON IMPLEMENTACIÓN POR DEFECTO
    # ============================================
    
    def predict(self, x):
        return self.forward(x)
    
    def zero_grad(self):
        """Resetea gradientes de TODOS los parámetros"""
        for p in self.parameters():
            p.grad = None
    
    def train(self, dataset, epochs: int = 100, lr: float = 0.01, 
              optimizer=None, loss_fn=None, verbose: bool = True):
        """
        Entrenamiento genérico.
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
                # Forward
                pred = self.forward(X)
                loss = loss_fn(pred, y)
                
                # Backward
                loss.backward()
                
                # Step
                optimizer.step()
                
                # Zero grad (todos los parámetros)
                self.zero_grad()
                
                total_loss += float(loss.data)
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
        """Guarda el modelo en disco"""
        import pickle
        state = self._get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 {self.name} guardado en {path}")
    
    def load(self, path: str):
        """Carga el modelo desde disco"""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self._set_state(state)
        self._trained = True
        print(f"📂 {self.name} cargado desde {path}")
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'version': self.version,
            'trained': self._trained,
            'history': self._training_history,
        }
    
    def _set_state(self, state: Dict[str, Any]):
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
        total = 0
        for p in self.parameters():
            if hasattr(p, 'data'):
                total += p.data.size
        return total
    
    def summary(self):
        print(f"🧠 {self.name} v{self.version}")
        print(f"   Entrenado: {'✅' if self._trained else '❌'}")
        print(f"   Parámetros: {self.num_parameters():,}")
        if self._training_history:
            print(f"   Épocas: {len(self._training_history)}")
            print(f"   Loss final: {self._training_history[-1]:.6f}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
