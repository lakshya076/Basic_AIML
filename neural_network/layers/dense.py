from .base import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size, initializer='random'):
        super().__init__()
        
        if initializer == 'xavier':
            limit = np.sqrt(6 / (input_size + output_size))
            w = np.random.uniform(-limit, limit, (output_size, input_size))
        elif initializer == 'he':
            w = np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        else: # Default random
            w = np.random.randn(output_size, input_size) * 0.01

        self.params = {
            'W': w,
            'b': np.zeros((output_size, 1))
        }
        self.grads = {
            'W': None,
            'b': None
        }

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.params['W'], self.input) + self.params['b']
        return self.output

    def backward(self, output_error):
        input_error = np.dot(self.params['W'].T, output_error)
        self.grads['W'] = np.dot(output_error, self.input.T)
        self.grads['b'] = np.sum(output_error, axis=1, keepdims=True)
        return input_error
