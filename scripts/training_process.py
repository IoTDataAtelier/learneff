import numpy as np
import os
from sklearn.model_selection import train_test_split
import scipy.stats as st
from classes.data_generation import DataGenerationStrategy
from classes.error_function import ErrorFunctionStrategy
from classes.model import ModelStrategy

#def mean_squared_error(y_true, y_pred):
#    return np.mean((y_true - y_pred) ** 2)

def error_train_test(y_train, y_test, y_pred, y_pred_test, e_phi: ErrorFunctionStrategy):
    e_train = e_phi.eval(y_true = y_train, y_pred = y_pred)
    e_test = e_phi.eval(y_true = y_test, y_pred = y_pred_test)

    return (e_train, e_test)

def pred_train_test(X_train, X_test, w, H: ModelStrategy):
    y_pred = H.pred(X = X_train, w = w)
    y_pred_test = H.pred(X = X_test, w = w)

    return (y_pred, y_pred_test)

def training_process(output_path: str, 
                    filepath: str, D: int, T: int, 
                    r_omega: DataGenerationStrategy, 
                    e_phi: ErrorFunctionStrategy, 
                    H: ModelStrategy, lr: float = 0.05):
    """
    Training process for synthetic dataset.
    Saves weights, training error, and test error over time.
    Returns only the weight matrix W (D x T).
    """
    import fastavro

    # --- Load dataset from Avro ---
    X, y = [], []
    with open(filepath, "rb") as f:
        reader = fastavro.reader(f)
        for record in reader:
            X.append(record["features"])
            y.append(record["target"])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Initialize weights ---
    w = r_omega.gen(D = D)

    # --- Initialize error vectors and Matrix W ---
    W = np.zeros((D, T))
    e_train = np.zeros(T)
    e_test = np.zeros(T)

    # --- Initial error ---
    y_pred, y_pred_test = pred_train_test(X_train, X_test, w, H)

    e_train[0], e_test[0] = error_train_test(y_train, y_test, y_pred, y_pred_test, e_phi)

    # --- Store initial weights ---
    W[:, 0] = w.flatten()

    for t in range(1, T):
        # Gradient descent update
        grad = -(2 / len(X_train)) * X_train.T @ (y_train - y_pred)
        w -= lr * grad

        # Predictions
        y_pred, y_pred_test = pred_train_test(X_train, X_test, w, H)

        # Errors
        e_train[t], e_test[t] = error_train_test(y_train, y_test, y_pred, y_pred_test, e_phi)

        # Store weights
        W[:, t] = w.flatten()

    # --- Save results ---
    np.save(os.path.join(output_path, "weights_over_time.npy"), W)
    np.save(os.path.join(output_path, "train_errors.npy"), e_train)
    np.save(os.path.join(output_path, "test_errors.npy"), e_test)

    # Return only W for graph generation step
    return W