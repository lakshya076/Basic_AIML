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


    def train(self, x_train, y_train, epochs, batch_size=32, verbose=True, x_val=None, y_val=None, save_path=None):
        # Determine if it's CNN or ANN
        is_cnn = (x_train.ndim == 4)
        n_samples = x_train.shape[0] if is_cnn else x_train.shape[1]
        
        # Set default save_path based on model type if not provided
        if save_path is None:
            from .layers import Conv2D
            has_cnn = any(isinstance(l, Conv2D) for l in self.layers)
            save_path = 'cnn_checkpoint.pkl' if has_cnn else 'best_model.pkl'

        best_acc = -1
        history = {'loss': [], 'val_acc': []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices] if is_cnn else x_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size] if is_cnn else x_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                if is_cnn:
                    batch_loss = 0
                    accumulated_grads = [ {k: np.zeros_like(v) for k, v in l.params.items()} for l in self.layers ]
                    
                    for j in range(x_batch.shape[0]):
                        sample_x = x_batch[j]
                        sample_y = y_batch[:, j:j+1]
                        
                        output = self.predict(sample_x)
                        error = self.loss.backward(sample_y, output)
                        
                        for l_idx in reversed(range(len(self.layers))):
                            error = self.layers[l_idx].backward(error)
                            # Accumulate gradients
                            for k in self.layers[l_idx].params:
                                accumulated_grads[l_idx][k] += self.layers[l_idx].grads[k]
                        
                        batch_loss += self.loss.forward(sample_y, output)
                    
                    # Set the averaged gradients back to layers for the optimizer
                    for l_idx, layer in enumerate(self.layers):
                        for k in layer.params:
                            layer.grads[k] = accumulated_grads[l_idx][k] / x_batch.shape[0]
                    
                    epoch_loss += batch_loss / x_batch.shape[0]
                else:
                    # Standard ANN batch processing
                    output = self.predict(x_batch)
                    error = self.loss.backward(y_batch, output)
                    for l_idx in reversed(range(len(self.layers))):
                        error = self.layers[l_idx].backward(error)
                    epoch_loss += self.loss.forward(y_batch, output)

                # Update weights once per batch
                self.optimizer.update(self.layers)

            avg_loss = epoch_loss / (n_samples / batch_size)
            history['loss'].append(avg_loss)
            
            val_info = ""
            if x_val is not None and y_val is not None:
                # Handle validation
                val_acc = self.evaluate(x_val, y_val)
                history['val_acc'].append(val_acc)
                val_info = f", val_acc={val_acc:.4f}"
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save_weights(save_path)
                    val_info += " (Best till now)\n"

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}{val_info}")
        
        return history


    def evaluate(self, x, y):
        # Keeps existing evaluation for backward compatibility in training loop
        is_cnn = (x.ndim == 4)
        if is_cnn:
            predictions = []
            for i in range(x.shape[0]):
                pred = self.predict(x[i])
                predictions.append(np.argmax(pred))
            targets = np.argmax(y, axis=0)
            return np.sum(np.array(predictions) == targets) / targets.size
        else:
            output = self.predict(x)
            # If output is 1 neuron, it might be regression or binary classification
            if output.shape[0] == 1:
                # Basic check: if y is only 0/1 it's likely binary class, else regression
                if np.all(np.isin(y, [0, 1])):
                    predictions = (output > 0.5).astype(int)
                    return np.mean(predictions == y)
                else:
                    # For regression, evaluate accuracy doesn't make sense, return -MSE
                    return -np.mean((output - y)**2)

            predictions = np.argmax(output, axis=0)
            targets = np.argmax(y, axis=0)
            return np.sum(predictions == targets) / targets.size


    def get_metrics(self, x, y):
        from preprocessing import get_classification_report, confusion_matrix, mean_squared_error, r2_score
        
        output = self.predict(x) if x.ndim != 4 else None
        is_cnn = (x.ndim == 4)
        
        # Classification Case
        # Multi-class (Softmax) or Binary (Sigmoid with 0/1 target)
        if is_cnn or output.shape[0] > 1 or np.all(np.isin(y, [0, 1])):
            if is_cnn:
                predictions = []
                for i in range(x.shape[0]):
                    predictions.append(np.argmax(self.predict(x[i])))
                y_pred = np.array(predictions)
                y_true = np.argmax(y, axis=0)
            elif output.shape[0] > 1: # Multi-class ANN
                y_pred = np.argmax(output, axis=0)
                y_true = np.argmax(y, axis=0)
            else: # Binary ANN
                y_pred = (output > 0.5).astype(int).flatten()
                y_true = y.flatten().astype(int)
            
            report = get_classification_report(y_true, y_pred)
            report["Confusion Matrix"] = confusion_matrix(y_true, y_pred)
            return report
        
        # Regression Case
        else:
            y_true = y.flatten()
            y_pred = output.flatten()
            return {
                "MSE": mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred)
            }
