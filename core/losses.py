"""
Funciones de pérdida - Compatibles con Tensor V3.1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error Loss"""
    
    def forward(self, pred, target):
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        diff = pred - target
        # MSE = mean(diff^2)
        squared = diff.pow(2)
        loss = squared.mean()
        return loss
    
    def __repr__(self):
        return "MSELoss()"


class BCELoss(Module):
    """Binary Cross Entropy Loss"""
    
    def forward(self, pred, target):
        if not isinstance(pred, Tensor):
            pred = Tensor(pred)
        if not isinstance(target, Tensor):
            target = Tensor(target)
        
        eps = 1e-10
        # Loss = -mean(target * log(pred) + (1-target) * log(1-pred))
        pred_clipped = Tensor(np.clip(pred.data, eps, 1 - eps), requires_grad=False)
        
        # Crear loss manualmente con backward
        term1 = target.data * np.log(pred_clipped.data)
        term2 = (1 - target.data) * np.log(1 - pred_clipped.data)
        loss_data = -np.mean(term1 + term2)
        
        loss = Tensor(loss_data, requires_grad=True, _children=(pred,))
        
        def backward():
            if pred.requires_grad and loss.grad is not None:
                n = pred.data.size
                grad = -(target.data / (pred.data + eps) - (1 - target.data) / (1 - pred.data + eps)) / n
                pred.grad = grad if pred.grad is None else pred.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "BCELoss()"


class CrossEntropyLoss(Module):
    """Cross Entropy Loss (con softmax implícito)"""
    
    def forward(self, logits, targets):
        if not isinstance(logits, Tensor):
            logits = Tensor(logits)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
        
        # Softmax
        max_val = np.max(logits.data, axis=1, keepdims=True)
        exp_data = np.exp(logits.data - max_val)
        probs = exp_data / (np.sum(exp_data, axis=1, keepdims=True) + 1e-10)
        
        # One-hot encoding
        if len(targets.data.shape) == 1:
            targets_onehot = np.zeros((targets.data.shape[0], logits.data.shape[1]))
            targets_onehot[np.arange(targets.data.shape[0]), targets.data.astype(int)] = 1
        else:
            targets_onehot = targets.data
        
        # Cross entropy
        eps = 1e-10
        loss_data = -np.mean(np.sum(targets_onehot * np.log(probs + eps), axis=1))
        
        loss = Tensor(loss_data, requires_grad=True, _children=(logits,))
        
        def backward():
            if logits.requires_grad and loss.grad is not None:
                grad = (probs - targets_onehot) / logits.data.shape[0]
                logits.grad = grad if logits.grad is None else logits.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "CrossEntropyLoss()"
