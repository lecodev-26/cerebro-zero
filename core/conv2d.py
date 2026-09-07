"""
Capa Convolucional 2D (Conv2D) con NumPy - Corregida
"""

import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Inicialización de pesos (Xavier)
        lim = np.sqrt(6.0 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.kernels = np.random.uniform(-lim, lim, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros((out_channels, 1))
        
        self.grad_kernels = None
        self.grad_bias = None
        self.input = None
        self.x_padded = None
    
    def forward(self, X):
        """
        X: (batch_size, in_channels, height, width)
        """
        self.input = X
        batch_size, in_channels, height, width = X.shape
        pad = self.padding
        
        # Aplicar padding
        if pad > 0:
            self.x_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
            height_pad = height + 2*pad
            width_pad = width + 2*pad
        else:
            self.x_padded = X
            height_pad = height
            width_pad = width
        
        # Dimensiones de salida
        out_height = (height_pad - self.kernel_size) // self.stride + 1
        out_width = (width_pad - self.kernel_size) // self.stride + 1
        
        # Inicializar salida
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Convolución
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        region = self.x_padded[b, :, h_start:h_end, w_start:w_end]
                        # Multiplicación elemento a elemento y suma (conversión a escalar)
                        valor = np.sum(region * self.kernels[c_out]) + self.bias[c_out, 0]
                        output[b, c_out, h, w] = valor
        
        return output
    
    def backward(self, grad_output, lr=0.01):
        """
        grad_output: gradiente de la pérdida con respecto a la salida
        """
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, height, width = self.input.shape
        
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_bias = np.zeros_like(self.bias)
        
        # Gradiente para bias
        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1)
        
        # Gradiente para kernels
        for b in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        region = self.x_padded[b, :, h_start:h_end, w_start:w_end]
                        self.grad_kernels[c_out] += grad_output[b, c_out, h, w] * region
        
        self.grad_kernels /= batch_size
        self.grad_bias /= batch_size
        
        # Actualizar pesos
        self.kernels -= lr * self.grad_kernels
        self.bias -= lr * self.grad_bias
        
        # Gradiente para la entrada (para propagar hacia atrás)
        grad_input = np.zeros((batch_size, in_channels, height, width))
        
        for b in range(batch_size):
            for c_in in range(in_channels):
                for h in range(height):
                    for w in range(width):
                        # Para cada posición, calcular contribución de todos los kernels
                        for c_out in range(out_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    h_start = h - kh + self.padding
                                    w_start = w - kw + self.padding
                                    if 0 <= h_start < out_height and 0 <= w_start < out_width:
                                        grad_input[b, c_in, h, w] += grad_output[b, c_out, h_start, w_start] * self.kernels[c_out, c_in, kh, kw]
        
        return grad_input
    
    def __repr__(self):
        return f"Conv2D({self.in_channels}→{self.out_channels}, kernel={self.kernel_size}, stride={self.stride})"
