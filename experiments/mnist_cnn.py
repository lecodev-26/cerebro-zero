"""
MNIST con CNN (NumPy puro - sin Tensor)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        lim = np.sqrt(6.0 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        self.kernels = np.random.uniform(-lim, lim, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros((out_channels, 1))
        self.grad_kernels = None
        self.grad_bias = None
        self.input = None
        self.x_padded = None
    
    def forward(self, X):
        self.input = X
        batch_size, in_channels, height, width = X.shape
        pad = self.padding
        
        if pad > 0:
            self.x_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
            height_pad = height + 2*pad
            width_pad = width + 2*pad
        else:
            self.x_padded = X
            height_pad = height
            width_pad = width
        
        out_height = (height_pad - self.kernel_size) // self.stride + 1
        out_width = (width_pad - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        region = self.x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h, w] = np.sum(region * self.kernels[c_out]) + self.bias[c_out, 0]
        
        return output
    
    def backward(self, grad_output, lr=0.01):
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, height, width = self.input.shape
        
        self.grad_kernels = np.zeros_like(self.kernels)
        self.grad_bias = np.zeros_like(self.bias)
        
        self.grad_bias = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1)
        
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
        
        self.kernels -= lr * self.grad_kernels
        self.bias -= lr * self.grad_bias
        
        grad_input = np.zeros((batch_size, in_channels, height, width))
        for b in range(batch_size):
            for c_in in range(in_channels):
                for h in range(height):
                    for w in range(width):
                        for c_out in range(out_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    h_start = h - kh + self.padding
                                    w_start = w - kw + self.padding
                                    if 0 <= h_start < out_height and 0 <= w_start < out_width:
                                        grad_input[b, c_in, h, w] += grad_output[b, c_out, h_start, w_start] * self.kernels[c_out, c_in, kh, kw]
        
        return grad_input

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
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        output[b, c, h, w] = region[max_idx]
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

class LinearNumpy:
    def __init__(self, in_features, out_features):
        lim = np.sqrt(6.0 / (in_features + out_features))
        self.weight = np.random.uniform(-lim, lim, (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        self.grad_weight = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, X):
        self.input = X
        return np.dot(X, self.weight) + self.bias
    
    def backward(self, grad_output, lr=0.01):
        self.grad_weight = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias
        
        return np.dot(grad_output, self.weight.T)

class ReLUNumpy:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0).astype(float)

class CNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 8, 3, 1, 1)
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(8, 16, 3, 1, 1)
        self.pool2 = MaxPool2D(2, 2)
        self.fc1 = LinearNumpy(16 * 7 * 7, 32)
        self.fc2 = LinearNumpy(32, 10)
        self.relu = ReLUNumpy()
        
        self.layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.fc1, self.relu, self.fc2]
    
    def forward(self, X):
        X = X.reshape(-1, 1, 28, 28)
        X = self.conv1.forward(X)
        X = self.pool1.forward(X)
        X = self.conv2.forward(X)
        X = self.pool2.forward(X)
        X = X.reshape(X.shape[0], -1)  # Flatten
        X = self.fc1.forward(X)
        X = self.relu.forward(X)
        X = self.fc2.forward(X)
        # Softmax
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def backward(self, grad_output, lr=0.01):
        grad = self.fc2.backward(grad_output, lr)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = grad.reshape(-1, 16, 7, 7)
        grad = self.pool2.backward(grad)
        grad = self.conv2.backward(grad, lr)
        grad = self.pool1.backward(grad)
        grad = self.conv1.backward(grad, lr)
        return grad
    
    def train_step(self, X, y, lr=0.01):
        output = self.forward(X)
        loss = -np.mean(np.sum(y * np.log(output + 1e-10), axis=1))
        grad = output - y
        self.backward(grad, lr)
        return loss
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
    
    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def guardar(self, nombre="cnn_mnist.pkl"):
        with open(nombre, 'wb') as f:
            data = {}
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'kernels'):
                    data[f'conv{i}_kernels'] = layer.kernels
                    data[f'conv{i}_bias'] = layer.bias
                elif hasattr(layer, 'weight'):
                    data[f'fc{i}_weight'] = layer.weight
                    data[f'fc{i}_bias'] = layer.bias
            pickle.dump(data, f)
        print(f"💾 CNN guardada en {nombre}")

def entrenar_mnist():
    print("🧠 ENTRENANDO CNN CON MNIST")
    print("="*40)
    
    try:
        X_train = np.load('../modelos_guardados/X_train.npy')
        y_train = np.load('../modelos_guardados/y_train.npy')
        X_test = np.load('../modelos_guardados/X_test.npy')
        y_test = np.load('../modelos_guardados/y_test.npy')
    except:
        print("❌ No se encontraron datos MNIST.")
        return
    
    # Usar 5000 ejemplos
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    y_onehot = np.zeros((len(y_train), 10))
    y_onehot[np.arange(len(y_train)), y_train] = 1
    
    cnn = CNN()
    
    print(f"\n📊 Datos: {len(X_train)} ejemplos")
    
    print("\n🏋️ ENTRENANDO...")
    batch_size = 64
    epochs = 10
    lr = 0.01
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_onehot[indices]
        
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            loss = cnn.train_step(X_batch, y_batch, lr)
            total_loss += loss
        
        acc_train = cnn.accuracy(X_train, y_train)
        acc_test = cnn.accuracy(X_test[:1000], y_test[:1000])
        print(f"   Epoch {epoch+1}/{epochs}: loss = {total_loss/(len(X_train)//batch_size):.4f}, "
              f"train_acc = {acc_train:.4f}, test_acc = {acc_test:.4f}")
    
    print("\n✅ ENTRENAMIENTO COMPLETADO!")
    cnn.guardar("../modelos_guardados/cnn_mnist.pkl")

if __name__ == "__main__":
    entrenar_mnist()
