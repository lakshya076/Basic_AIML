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
from neural_network.activations import relu, relu_prime, sigmoid, sigmoid_prime, softmax
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import Adam

def load_data():
    dataset_path = os.path.join(ROOT_DIR, "datasets", "nn", "train.csv")
    df = pd.read_csv(dataset_path)
    data = np.array(df)
    data_train = data[0:1500].T
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.
    
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    return X_train, one_hot(Y_train)

def run_experiment():
    X_train, Y_train = load_data()
    epochs = 20
    
    # Deep Sigmoid Model (Suffers from Vanishing Gradient)
    print("Training Deep Sigmoid Model...")
    sigmoid_model = Sequential([
        Dense(784, 128), Activation(sigmoid, sigmoid_prime),
        Dense(128, 64), Activation(sigmoid, sigmoid_prime),
        Dense(64, 32), Activation(sigmoid, sigmoid_prime),
        Dense(32, 16), Activation(sigmoid, sigmoid_prime),
        Dense(16, 10), Activation(softmax, lambda x: 1)
    ])
    sigmoid_model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.01))
    history_sigmoid = sigmoid_model.train(X_train, Y_train, epochs=epochs, batch_size=64, verbose=False)
    
    # Deep ReLU Model (Mitigates Vanishing Gradient)
    print("Training Deep ReLU Model...")
    relu_model = Sequential([
        Dense(784, 128), Activation(relu, relu_prime),
        Dense(128, 64), Activation(relu, relu_prime),
        Dense(64, 32), Activation(relu, relu_prime),
        Dense(32, 16), Activation(relu, relu_prime),
        Dense(16, 10), Activation(softmax, lambda x: 1)
    ])
    relu_model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.01))
    history_relu = relu_model.train(X_train, Y_train, epochs=epochs, batch_size=64, verbose=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_sigmoid['loss'], label='Deep Sigmoid (Vanishing Gradient)')
    plt.plot(history_relu['loss'], label='Deep ReLU')
    plt.title('Vanishing Gradient Research: Sigmoid vs ReLU in Deep Nets')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(SCRIPT_DIR, 'vanishing_gradient.png')
    plt.savefig(save_path)
    print(f"Research complete. Plot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    run_experiment()
