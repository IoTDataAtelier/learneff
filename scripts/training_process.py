import numpy as np
import os
from sklearn.model_selection import train_test_split
import scipy.stats as st
from classes.data_generation import DataGenerationStrategy

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def training_process(output_path: str, filepath: str, D: int, T: int, r_omega: DataGenerationStrategy, lr: float = 0.05):
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

    W = np.zeros((D, T))
    e_train = np.zeros(T)
    e_test = np.zeros(T)

    for t in range(T):
        # Predictions
        y_pred = X_train @ w
        y_pred_test = X_test @ w

        # Errors
        e_train[t] = mean_squared_error(y_train, y_pred)
        e_test[t] = mean_squared_error(y_test, y_pred_test)

        # Store weights
        W[:, t] = w.flatten()

        # Gradient descent update
        grad = -(2 / len(X_train)) * X_train.T @ (y_train - y_pred)
        w -= lr * grad

    # --- Save results ---
    np.save(os.path.join(output_path, "weights_over_time.npy"), W)
    np.save(os.path.join(output_path, "train_errors.npy"), e_train)
    np.save(os.path.join(output_path, "test_errors.npy"), e_test)

    # Return only W for graph generation step
    return W