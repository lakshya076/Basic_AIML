import numpy as np
import pandas as pd
from neural_network import Sequential
from neural_network.layers import Dense, Activation, Conv2D, Flatten, MaxPool2D
from neural_network.activations import relu, relu_prime, softmax, softmax_prime, sigmoid, sigmoid_prime
from neural_network.losses import CategoricalCrossEntropy
from neural_network.optimizers import Adam, SGD


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def load_data():
    df = pd.read_csv("train.csv")
    data = np.array(df)
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    
    return X_train, one_hot(Y_train), X_dev, one_hot(Y_dev)


def train_ann():
    print("Training ANN...")
    X_train, Y_train, X_dev, Y_dev = load_data()
    
    model = Sequential([
        Dense(784, 128),
        Activation(relu, relu_prime),
        Dense(128, 64),
        Activation(relu, relu_prime),
        Dense(64, 10),
        Activation(softmax, lambda x: 1)
    ])
    
    model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    model.train(
        X_train, Y_train, 
        epochs=10, 
        batch_size=64, 
        x_val=X_dev, 
        y_val=Y_dev, 
        save_path='checkpoint_model.pkl')
    
    model.load_weights('checkpoint_model.pkl')
    predictions = model.predict(X_dev)
    accuracy = np.sum(np.argmax(predictions, axis=0) == np.argmax(Y_dev, axis=0)) / Y_dev.shape[1]
    print(f"ANN Dev Accuracy: {accuracy}")


def train_cnn():
    print("\nTraining CNN...")
    X_train, Y_train, X_dev, Y_dev = load_data()
    
    # Reshape for CNN: (samples, depth, height, width)
    X_train_cnn = X_train.T.reshape(-1, 1, 28, 28)
    X_dev_cnn = X_dev.T.reshape(-1, 1, 28, 28)
    
    # Tiny subset for speed
    train_size, val_size = 100, 20
    X_train_small = X_train_cnn[:train_size]
    Y_train_small = Y_train[:, :train_size]
    X_dev_small = X_dev_cnn[:val_size]
    Y_dev_small = Y_dev[:, :val_size]
    
    model = Sequential([
        Conv2D((1, 28, 28), kernel_size=3, depth=4),
        Activation(relu, relu_prime),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(4 * 13 * 13, 10),
        Activation(softmax, lambda x: 1)
    ])
    
    model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(learning_rate=0.001))
    model.train(
        X_train_small, Y_train_small, 
        epochs=5, 
        batch_size=10, 
        x_val=X_dev_small, 
        y_val=Y_dev_small,
        save_path='cnn_checkpoint.pkl'
    )
    
    model.load_weights('cnn_checkpoint.pkl')
    accuracy = model.evaluate(X_dev_small, Y_dev_small)
    print(f"CNN Subset Accuracy: {accuracy}")


if __name__ == "__main__":
    train_ann()
    train_cnn()
