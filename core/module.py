"""
Módulo base para redes neuronales
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tensor import Tensor

class Module:
    """Módulo base para todas las capas y modelos"""
    
    def __init__(self):
        self._params = []
        self._children = []
        self._name = self.__class__.__name__
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclases deben implementar forward()")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """Devuelve todos los parámetros del módulo y sus hijos"""
        params = []
        # Parámetros propios
        for param in self._params:
            params.append(param)
        # Parámetros de los hijos
        for child in self._children:
            params.extend(child.parameters())
        return params
    
    def zero_grad(self):
        """Pone a cero los gradientes de todos los parámetros"""
        for param in self.parameters():
            param.grad = None
    
    def add_param(self, param):
        """Añade un parámetro al módulo"""
        self._params.append(param)
    
    def add_module(self, name, module):
        """Añade un submódulo"""
        setattr(self, name, module)
        self._children.append(module)
    
    def __repr__(self):
        return f"{self._name}()"
