import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        lim = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-lim, lim, (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        self.grad_weight = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad):
        self.grad_weight = np.dot(self.input.T, grad)
        self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weight.T)
    
    def update(self, lr):
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias
    
    def __repr__(self):
        return f"Linear({self.in_features} → {self.out_features})"
