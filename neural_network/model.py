import numpy as np
import pickle

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

    def save_weights(self, filepath):
        weights = []
        for layer in self.layers:
            weights.append(layer.params)
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        for i, layer in enumerate(self.layers):
            layer.params = weights[i]
        print(f"Model weights loaded from {filepath}")

    def train(self, x_train, y_train, epochs, batch_size=32, verbose=True, x_val=None, y_val=None, save_path='checkpoint.pkl'):
        n_samples = x_train.shape[1]
        best_acc = -1

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

            avg_loss = epoch_loss / (n_samples / batch_size)
            
            val_info = ""
            if x_val is not None and y_val is not None:
                val_output = self.predict(x_val)
                # Assuming classification for accuracy
                predictions = np.argmax(val_output, axis=0)
                targets = np.argmax(y_val, axis=0)
                acc = np.sum(predictions == targets) / targets.size
                val_info = f", val_acc={acc:.4f}"
                
                if acc > best_acc:
                    best_acc = acc
                    self.save_weights(save_path)
                    val_info += " (Best!)"

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}{val_info}")
