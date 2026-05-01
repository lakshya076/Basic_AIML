import numpy as np
from neural_network import Sequential
from neural_network.layers import Dense
from neural_network.losses import MSE
from neural_network.optimizers import SGD


class LinearRegression:
    def __init__(self, input_dim, optimizer=None):
        self.model = Sequential([
            Dense(input_dim, 1)
        ])
        self.loss = MSE()
        self.optimizer = optimizer if optimizer else SGD(learning_rate=0.01)
        self.model.compile(self.loss, self.optimizer)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True):
        # Sequential.train expects X to be (features, samples)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        self.model.train(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_metrics(self, X, y):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        return self.model.get_metrics(X, y)
