import numpy as np
from neural_network import Sequential
from neural_network.layers import Dense, Activation
from neural_network.activations import sigmoid, sigmoid_prime
from neural_network.losses import BinaryCrossEntropy
from neural_network.optimizers import SGD


class LogisticRegression:
    def __init__(self, input_dim, optimizer=None):
        self.model = Sequential([
            Dense(input_dim, 1),
            Activation(sigmoid, sigmoid_prime)
        ])
        self.loss = BinaryCrossEntropy()
        self.optimizer = optimizer if optimizer else SGD(learning_rate=0.1)
        self.model.compile(self.loss, self.optimizer)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        self.model.train(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_classes(self, X, threshold=0.5):
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def get_metrics(self, X, y):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        return self.model.get_metrics(X, y)
