import numpy as np
import pandas as pd
from neural_network import Sequential
from neural_network.layers import Dense, Activation
from neural_network.activations import relu, relu_prime, softmax


def generate_predictions():
    print("Loading test data...")
    test_df = pd.read_csv("test.csv")
    X_test = np.array(test_df).T / 255.
    
    model = Sequential([
        Dense(784, 128),
        Activation(relu, relu_prime),
        Dense(128, 64),
        Activation(relu, relu_prime),
        Dense(64, 10),
        Activation(softmax, lambda x: 1)
    ])
    
    try:
        model.load_weights('checkpoint_model.pkl')
    except FileNotFoundError:
        print("Error: 'checkpoint_model.pkl' not found. Please run train_modular.py first.")
        return
    
    print("Generating predictions...")
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=0)
    
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predicted_labels) + 1),
        "Label": predicted_labels
    })
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    generate_predictions()
