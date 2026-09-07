"""
Funciones de pérdida (loss functions) - Completas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.module import Module
from core.tensor import Tensor

class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        n = target.data.size
        loss_data = np.mean(diff.data ** 2)
        
        loss = Tensor(loss_data, requires_grad=True)
        
        def backward():
            if pred.requires_grad:
                grad = 2 * diff.data / n
                pred.grad = grad if pred.grad is None else pred.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "MSELoss()"

class BCELoss(Module):
    def forward(self, pred, target):
        eps = 1e-10
        self.pred = pred
        self.target = target
        pred_clip = np.clip(pred.data, eps, 1 - eps)
        loss_data = -np.mean(target.data * np.log(pred_clip) + (1 - target.data) * np.log(1 - pred_clip))
        
        loss = Tensor(loss_data, requires_grad=True)
        
        def backward():
            if pred.requires_grad:
                grad = -(target.data / (pred.data + eps) - (1 - target.data) / (1 - pred.data + eps)) / pred.data.size
                pred.grad = grad if pred.grad is None else pred.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "BCELoss()"

class CrossEntropyLoss(Module):
    """Cross Entropy Loss (con softmax implícito)"""
    
    def forward(self, logits, targets):
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
            if logits.requires_grad:
                grad = (probs - targets_onehot) / logits.data.shape[0]
                if logits.grad is None:
                    logits.grad = grad
                else:
                    logits.grad = logits.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "CrossEntropyLoss()"

class L1Loss(Module):
    def forward(self, pred, target):
        diff = pred - target
        n = target.data.size
        loss_data = np.mean(np.abs(diff.data))
        
        loss = Tensor(loss_data, requires_grad=True)
        
        def backward():
            if pred.requires_grad:
                grad = np.sign(diff.data) / n
                pred.grad = grad if pred.grad is None else pred.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return "L1Loss()"

class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, target):
        diff = pred - target
        abs_diff = np.abs(diff.data)
        n = target.data.size
        loss_data = np.mean(np.where(abs_diff < self.beta, 
                                     0.5 * diff.data**2 / self.beta,
                                     abs_diff - 0.5 * self.beta))
        
        loss = Tensor(loss_data, requires_grad=True)
        
        def backward():
            if pred.requires_grad:
                grad = np.where(abs_diff < self.beta, diff.data / self.beta, np.sign(diff.data)) / n
                pred.grad = grad if pred.grad is None else pred.grad + grad
        
        loss._backward = backward
        return loss
    
    def __repr__(self):
        return f"SmoothL1Loss(beta={self.beta})"
