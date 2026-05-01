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

- **Regression Support:**
  - `LinearRegression`: Standard regression using MSE.
  - `LogisticRegression`: Binary classification using Binary Cross Entropy.
- **Standardized Modules:** Optimizers and loss functions are shared between Neural Networks and Regression models for consistency.

## 📂 Project Structure

```text
nn/
├── model.py            # Main Sequential model logic
├── layers/             # Dense, Conv2D, MaxPool2D, Flatten, Activation
├── activations/        # ReLU, Sigmoid, Tanh, Softmax
├── losses/             # MSE, CategoricalCrossEntropy, BinaryCrossEntropy
└── optimizers/         # SGD, Adam

regression/
├── linear.py           # Linear Regression implementation
└── logistic.py         # Logistic Regression implementation

preprocessing/
├── __init__.py
├── data.py             # train_test_split implementation
└── metrics.py          # Accuracy, Precision, Recall, Confusion Matrix
```

## 📊 Datasets

The following datasets are used to train and evaluate the models in this project:

| Model Type | Dataset Name | File Path |
| :--- | :--- | :--- |
| **ANN / CNN** | MNIST Digit Recognizer | `datasets/nn/train.csv` (Labeled)<br>`datasets/nn/test.csv` (Unlabeled) |
| **Linear Regression** | Boston Housing Data | `datasets/regression/linear/HousingData.csv` |
| **Logistic Regression** | Loan Approval Data | `datasets/regression/logistic/loan_data.csv` |

## 📈 Metrics & Evaluation

Each model type calculates specific performance indicators through the `preprocessing/metrics.py` module:

### Regression (Linear)

- **MSE (Mean Squared Error):** Measures the average squared difference between predicted and actual values.
- **R² Score (Coefficient of Determination):** Represents the proportion of variance for the dependent variable that's explained by the model.

### Classification (Logistic, ANN, CNN)

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** Accuracy of positive predictions.
- **Recall:** Ability of the model to find all positive instances.
- **Confusion Matrix:** A visual heatmap (generated via Matplotlib) showing True Positives, True Negatives, False Positives, and False Negatives.

## 🏗 Architecture

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

## How to Use ANN/CNN

Either run the `main.ipynb` notebook or:

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

## How to Use Linear and Logistic Regression

Run the `regression_sample.ipynb` notebook.
