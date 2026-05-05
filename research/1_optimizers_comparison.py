import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from neural_network import Sequential
from neural_network.layers import Dense, Activation
from neural_network.activations import relu, relu_prime, softmax
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import Adam, SGD

def load_data():
    dataset_path = os.path.join(ROOT_DIR, "datasets", "nn", "train.csv")
    df = pd.read_csv(dataset_path)
    data = np.array(df)
    m, n = data.shape
    np.random.shuffle(data)

    data_train = data[0:2000].T # Smaller subset for research speed
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.

    data_val = data[2000:3000].T
    Y_val = data_val[0]
    X_val = data_val[1:n] / 255.
    
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    return X_train, one_hot(Y_train), X_val, one_hot(Y_val)

def run_experiment():
    X_train, Y_train, X_val, Y_val = load_data()
    
    epochs = 15
    batch_size = 64
    
    # SGD Model
    print("Training with SGD...")
    model_sgd = Sequential([
        Dense(784, 128), Activation(relu, relu_prime),
        Dense(128, 10), Activation(softmax, lambda x: 1)
    ])
    model_sgd.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(learning_rate=0.1))
    history_sgd = model_sgd.train(X_train, Y_train, epochs=epochs, batch_size=batch_size, x_val=X_val, y_val=Y_val, verbose=False)
    
    # Adam Model
    print("Training with Adam...")
    model_adam = Sequential([
        Dense(784, 128), Activation(relu, relu_prime),
        Dense(128, 10), Activation(softmax, lambda x: 1)
    ])
    model_adam.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    history_adam = model_adam.train(X_train, Y_train, epochs=epochs, batch_size=batch_size, x_val=X_val, y_val=Y_val, verbose=False)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_sgd['loss'], label='SGD (LR=0.1)')
    plt.plot(history_adam['loss'], label='Adam (LR=0.001)')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history_sgd['val_acc'], label='SGD (LR=0.1)')
    plt.plot(history_adam['val_acc'], label='Adam (LR=0.001)')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(SCRIPT_DIR, 'optimizers_comparison.png')
    plt.savefig(save_path)
    print(f"Research complete. Plot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    run_experiment()
