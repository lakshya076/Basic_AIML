import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, layers):
        for layer in layers:
            for param_name in layer.params:
                layer.params[param_name] -= self.learning_rate * layer.grads[param_name]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if not layer.params:
                continue
            
            if i not in self.m:
                self.m[i] = {k: 0 for k in layer.params}
                self.v[i] = {k: 0 for k in layer.params}

            for k in layer.params:
                self.m[i][k] = self.beta1 * self.m[i][k] + (1 - self.beta1) * layer.grads[k]
                self.v[i][k] = self.beta2 * self.v[i][k] + (1 - self.beta2) * (layer.grads[k]**2)

                m_hat = self.m[i][k] / (1 - self.beta1**self.t)
                v_hat = self.v[i][k] / (1 - self.beta2**self.t)

                layer.params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
