"""
Max Pooling 2D con NumPy
"""

import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.max_indices = None
    
    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        region = X[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        output[b, c, h, w] = max_val
                        self.max_indices[b, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, grad_output):
        batch_size, channels, out_height, out_width = grad_output.shape
        _, _, height, width = self.input.shape
        grad_input = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_idx, w_idx = self.max_indices[b, c, h, w]
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, h, w]
        
        return grad_input
    
    def __repr__(self):
        return f"MaxPool2D(kernel={self.kernel_size}, stride={self.stride})"
