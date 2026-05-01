import numpy as np
import matplotlib.pyplot as plt

# Regression Metrics

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


# Classification Metrics
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Computes confusion matrix for binary or multi-class classification.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2: # Binary
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    else: # Multi-class (Macro)
        precisions = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return np.mean(precisions)


def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2: # Binary
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    else: # Multi-class (Macro)
        recalls = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return np.mean(recalls)


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true.flatten(), y_pred.flatten()]))
        # For binary, use friendly names
        if len(labels) == 2 and set(labels) == {0, 1}:
            labels = ['Negative (0)', 'Positive (1)']
        
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def get_classification_report(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    is_binary = len(np.unique(np.concatenate([y_true, y_pred]))) <= 2
    
    suffix = "" if is_binary else " (Macro)"
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        f"Precision{suffix}": precision_score(y_true, y_pred),
        f"Recall{suffix}": recall_score(y_true, y_pred)
    }
    return metrics
