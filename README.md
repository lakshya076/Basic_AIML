# Neural Network from Scratch

A modular and extensible deep learning framework built entirely from scratch using only Python, NumPy, and SciPy. This project demonstrates the fundamental principles of backpropagation, gradient descent, and convolutional neural networks without relying on high-level libraries like PyTorch or TensorFlow.

## Features

- **Modular Design:** Separate components for Layers, Activations, Losses, and Optimizers.
- **Sequential API:** Easy model construction using a Keras-like `Sequential` class.
- **Layer Support:**
  - `Dense`: Fully connected layers for standard ANNs.
  - `Conv2D`: Convolutional layers for spatial feature extraction.
  - `MaxPool2D`: Downsampling layers for CNNs.
  - `Flatten`: Transition layer from spatial to linear dimensions.
- **Advanced Optimizers:** Support for `SGD` and `Adam`.
- **Numerical Stability:** Stabilized Softmax and clipped Cross-Entropy implementations to prevent `NaN` or `inf` errors.
- **Automatic Checkpoints:** Automatically saves the best-performing model state based on validation accuracy during training.

## Project Structure

```text
neural_network/
├── model.py            # Main Sequential model logic
├── layers/             # Dense, Conv2D, MaxPool2D, Flatten, Activation
├── activations/        # ReLU, Sigmoid, Tanh, Softmax
├── losses/             # MSE, CategoricalCrossEntropy, BinaryCrossEntropy
└── optimizers/         # SGD, Adam
```

## Architecture

### Artificial Neural Network (ANN)

The default ANN configuration used in `train_modular.py` consists of:

1. Input Layer (784 features)
2. Dense Layer (128 neurons) + ReLU
3. Dense Layer (64 neurons) + ReLU
4. Dense Layer (10 neurons) + Softmax

### Convolutional Neural Network (CNN)

The demonstration CNN architecture consists of:

1. Conv2D (4 filters, 3x3 kernel) + ReLU
2. MaxPool2D (2x2 pool size)
3. Flatten
4. Dense (Output layer) + Softmax

## Important Note on CNN Performance

While the framework supports full CNN capabilities, the current implementation uses **pure NumPy loops** on the **CPU**.

- **Subset Training:** In `train_modular.py`, the CNN is trained on a **tiny subset (500 images)**. This is to ensure the script finishes in a reasonable time for demonstration purposes.
- **Bottleneck:** Without GPU acceleration (CUDA) or highly optimized C++ kernels (like `im2col`), spatial convolutions are computationally expensive in Python.
- **Full Training:** You can train on the full dataset by removing the subset slicing in `train_modular.py`, but expect it to take several hours per epoch.

## How to Use

### 1. Training

Run the modular training script to train both the ANN and a preview of the CNN:

```bash
python train_modular.py
```

This will generate `checkpoint_model.pkl` (for ANN) and `cnn_checkpoint.pkl` (for CNN).

### 2. Predictions

To generate predictions for the unlabeled `test.csv` using the best saved ANN model:

```bash
python test_predictions.py
```

This will output `submission.csv`.

## Accuracy

The standard ANN architecture consistently achieves **>97% accuracy** on the MNIST development set within 10 epochs using the Adam optimizer.
