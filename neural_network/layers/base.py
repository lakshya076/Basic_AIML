import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.params = {}
        self.grads = {}

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error):
        raise NotImplementedError
