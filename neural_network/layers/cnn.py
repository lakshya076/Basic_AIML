from .base import Layer
import numpy as np
from scipy import signal

class Conv2D(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        
        self.params = {
            'K': np.random.randn(*self.kernels_shape) * 0.1,
            'b': np.random.randn(depth, *self.output_shape[1:]) * 0.1
        }
        self.grads = {
            'K': np.zeros(self.kernels_shape),
            'b': np.zeros((depth, *self.output_shape[1:]))
        }

    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.params['b'])
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.params['K'][i, j], "valid")
        return self.output

    def backward(self, output_error):
        self.grads['K'] = np.zeros(self.kernels_shape)
        self.grads['b'] = output_error
        input_error = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.grads['K'][i, j] = signal.correlate2d(self.input[j], output_error[i], "valid")
                input_error[j] += signal.convolve2d(output_error[i], self.params['K'][i, j], "full")

        return input_error

class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1).T

    def backward(self, output_error):
        return output_error.T.reshape(self.input_shape)

class MaxPool2D(Layer):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input_data):
        self.input = input_data
        self.depth, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        
        self.output = np.zeros((self.depth, self.output_height, self.output_width))
        self.arg_max = np.zeros((self.depth, self.output_height, self.output_width, 2), dtype=int)

        for d in range(self.depth):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    h_start = i * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = j * self.pool_size
                    w_end = w_start + self.pool_size
                    
                    region = input_data[d, h_start:h_end, w_start:w_end]
                    self.output[d, i, j] = np.max(region)
                    
                    # Store relative index for backward pass
                    max_idx = np.unravel_index(np.argmax(region), region.shape)
                    self.arg_max[d, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return self.output

    def backward(self, output_error):
        input_error = np.zeros(self.input.shape)
        for d in range(self.depth):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    h, w = self.arg_max[d, i, j]
                    input_error[d, h, w] = output_error[d, i, j]
        return input_error
