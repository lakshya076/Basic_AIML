import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError
