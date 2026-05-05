import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from neural_network import Sequential
from neural_network.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten
from neural_network.activations import relu, relu_prime, softmax
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import Adam

def load_data():
    dataset_path = os.path.join(ROOT_DIR, "datasets", "nn", "train.csv")
    df = pd.read_csv(dataset_path)
    data = np.array(df)
    X = data[:, 1:].reshape(-1, 1, 28, 28) / 255.
    Y = data[:, 0]
    
    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    return X[:300], one_hot(Y[:300]), X[300:400], one_hot(Y[300:400])

def shift_images(X, pixels=3):
    """Shifts images to test spatial invariance."""
    shifted = np.zeros_like(X)
    for i in range(X.shape[0]):
        # Shift images right by 'pixels'
        shifted[i, 0, :, pixels:] = X[i, 0, :, :-pixels]
    return shifted

def run_experiment():
    X_train, Y_train, X_test, Y_test = load_data()
    
    # Model with MaxPool
    print("Training CNN with MaxPool...")
    model_pool = Sequential([
        Conv2D((1, 28, 28), kernel_size=3, depth=4),
        Activation(relu, relu_prime),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(4 * 13 * 13, 10),
        Activation(softmax, lambda x: 1)
    ])
    model_pool.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.005))
    model_pool.train(X_train, Y_train, epochs=3, batch_size=10, verbose=True)
    
    # Model without MaxPool (just Flatten after Conv)
    print("\nTraining CNN without MaxPool...")
    model_no_pool = Sequential([
        Conv2D((1, 28, 28), kernel_size=3, depth=4),
        Activation(relu, relu_prime),
        Flatten(),
        Dense(4 * 26 * 26, 10),
        Activation(softmax, lambda x: 1)
    ])
    model_no_pool.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.005))
    model_no_pool.train(X_train, Y_train, epochs=3, batch_size=10, verbose=True)
    
    # Test on normal vs shifted images
    X_test_shifted = shift_images(X_test, pixels=4)
    
    acc_pool_norm = model_pool.evaluate(X_test, Y_test)
    acc_pool_shift = model_pool.evaluate(X_test_shifted, Y_test)
    
    acc_no_pool_norm = model_no_pool.evaluate(X_test, Y_test)
    acc_no_pool_shift = model_no_pool.evaluate(X_test_shifted, Y_test)
    
    print("\nResults:")
    print(f"MaxPool Model: Normal Acc={acc_pool_norm:.4f}, Shifted Acc={acc_pool_shift:.4f}")
    print(f"No Pool Model:  Normal Acc={acc_no_pool_norm:.4f}, Shifted Acc={acc_no_pool_shift:.4f}")
    
    # Data for plot
    labels = ['Normal', 'Shifted']
    pool_scores = [acc_pool_norm, acc_pool_shift]
    no_pool_scores = [acc_no_pool_norm, acc_no_pool_shift]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, pool_scores, width, label='With MaxPool')
    ax.bar(x + width/2, no_pool_scores, width, label='No MaxPool')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Spatial Invariance: MaxPool vs No Pool')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    save_path = os.path.join(SCRIPT_DIR, 'cnn_invariance.png')
    plt.savefig(save_path)
    print(f"\nResearch complete. Plot saved as '{save_path}'")
    plt.show()

if __name__ == "__main__":
    run_experiment()
