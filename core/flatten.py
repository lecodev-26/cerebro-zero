import numpy as np

class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
    def __repr__(self):
        return "Flatten()"
