import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays or matrices into random train and test subsets.
    
    Args:
        X: pandas DataFrame or numpy array of features.
        y: pandas DataFrame/Series or numpy array of targets.
        test_size: float between 0 and 1, representing the proportion of the dataset to include in the test split.
        random_state: int for reproducibility.
        shuffle: Whether or not to shuffle the data before splitting.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    # Handle pandas DataFrames/Series vs numpy arrays
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
