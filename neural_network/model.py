import numpy as np

class Sequential:
    def __init__(self, layers=[], loss=None, optimizer=None):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, batch_size=32, verbose=True):
        n_samples = x_train.shape[1]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                output = self.predict(x_batch)
                error = self.loss.backward(y_batch, output)

                for layer in reversed(self.layers):
                    error = layer.backward(error)

                self.optimizer.update(self.layers)
                
                epoch_loss += self.loss.forward(y_batch, output)

            if verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, loss={epoch_loss / (n_samples / batch_size)}")
