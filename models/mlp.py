"""
MLP usando la interfaz Brain unificada
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from models.brain import Brain
from core.tensor import Tensor
from core.layers import Linear, Sequential
from core.activations import ReLU, Sigmoid, Tanh


class MLP(Brain):
    """
    Multilayer Perceptron usando la interfaz Brain.
    
    Ejemplo:
        model = MLP([2, 8, 1])
        model.forward(Tensor([[0, 1]]))
    """
    
    def __init__(self, layer_sizes, activation='relu', output_activation='sigmoid'):
        super().__init__(name=f"MLP{layer_sizes}")
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.output_activation_name = output_activation
        
        # Crear capas
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            # Añadir activación (excepto en la última capa)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(ReLU())
                elif activation == 'tanh':
                    layers.append(Tanh())
                elif activation == 'sigmoid':
                    layers.append(Sigmoid())
            else:
                # Activación de salida
                if output_activation == 'sigmoid':
                    layers.append(Sigmoid())
                elif output_activation == 'tanh':
                    layers.append(Tanh())
                elif output_activation == 'relu':
                    layers.append(ReLU())
                # Si es 'linear', no añadimos nada
        
        self.model = Sequential(*layers)
        self._trained = False
    
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def _get_state(self):
        """Guardar pesos"""
        state = super()._get_state()
        state['layer_sizes'] = self.layer_sizes
        state['activation'] = self.activation_name
        state['output_activation'] = self.output_activation_name
        state['weights'] = [
            {
                'weight': layer.weight.data.copy() if hasattr(layer, 'weight') else None,
                'bias': layer.bias.data.copy() if hasattr(layer, 'bias') and layer.bias is not None else None,
            }
            for layer in self.model.layers
        ]
        return state
    
    def _set_state(self, state):
        super()._set_state(state)
        self.layer_sizes = state.get('layer_sizes', self.layer_sizes)
        weights = state.get('weights', [])
        for layer, w in zip(self.model.layers, weights):
            if hasattr(layer, 'weight') and w['weight'] is not None:
                layer.weight.data = w['weight'].copy()
            if hasattr(layer, 'bias') and w['bias'] is not None and layer.bias is not None:
                layer.bias.data = w['bias'].copy()
    
    def __repr__(self):
        return f"MLP({self.layer_sizes})"


# Test
if __name__ == "__main__":
    print("🧪 PROBANDO MLP CON INTERFAZ BRAIN")
    print("="*50)
    
    # Crear MLP
    model = MLP([2, 8, 1])
    model.summary()
    
    # Forward
    x = Tensor([[0.0, 1.0]])
    y = model.forward(x)
    print(f"\n📊 Forward: {x.data} → {y.data}")
    
    # Predicción
    pred = model.predict(x)
    print(f"🔮 Predicción: {pred.data}")
    
    print("\n✅ MLP CON BRAIN FUNCIONANDO")
