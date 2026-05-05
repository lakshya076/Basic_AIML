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
    epochs = 10
    
    # Random Init
    print("Training with Random Init...")
    model_rand = Sequential([
        Dense(784, 128, initializer='random'), Activation(relu, relu_prime),
        Dense(128, 10), Activation(softmax, lambda x: 1)
    ])
    model_rand.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    history_rand = model_rand.train(X_train, Y_train, epochs=epochs, batch_size=64, verbose=False)
    
    # Xavier Init
    print("Training with Xavier Init...")
    model_xavier = Sequential([
        Dense(784, 128, initializer='xavier'), Activation(relu, relu_prime),
        Dense(128, 10), Activation(softmax, lambda x: 1)
    ])
    model_xavier.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    history_xavier = model_xavier.train(X_train, Y_train, epochs=epochs, batch_size=64, verbose=False)

    # He Init
    print("Training with He Init...")
    model_he = Sequential([
        Dense(784, 128, initializer='he'), Activation(relu, relu_prime),
        Dense(128, 10), Activation(softmax, lambda x: 1)
    ])
    model_he.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    history_he = model_he.train(X_train, Y_train, epochs=epochs, batch_size=64, verbose=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_rand['loss'], label='Random Initialization')
    plt.plot(history_xavier['loss'], label='Xavier Initialization')
    plt.plot(history_he['loss'], label='He Initialization')
    plt.title('Weight Initialization Study: Impact on Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(SCRIPT_DIR, 'initialization_study.png')
    plt.savefig(save_path)
    print(f"Research complete. Plot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    run_experiment()
