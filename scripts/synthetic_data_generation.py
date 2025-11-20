import numpy as np
from lib.functions import save_dataset_avro
from classes.data_generation import DataGenerationStrategy
import scipy.stats as st
import random

#def f_theta(N: int, D: int) -> np.ndarray:
#    return np.random.randn(N, D)


#def r_omega(D: int) -> np.ndarray:
    #return np.random.randn(D, 1)
#    v = np.array(st.norm.rvs(size = D)).T
#    return v.reshape(-1, 1)

#def g_lambda(X: np.ndarray, w: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
#    noise = noise_level * np.random.randn(X.shape[0], 1)
#    return X @ w + noise

def data_drop(X, drop_rate):
    size = X.shape[1]
    dp = random.sample(range(1, size), int((size - 1) * drop_rate))
    X_del = np.delete(X, dp, axis = 1)

    return X_del, dp

def synthetic_data_generation(output_path: str, 
                              f_theta: DataGenerationStrategy, 
                              r_omega: DataGenerationStrategy, 
                              g_lambda: DataGenerationStrategy,
                              cov: np.ndarray,
                              drop_w = None,
                              drop_data = None,
                              N: int = 200, D: int = 5, noise: float = 0.1
                              ):
    X = f_theta.gen(N = N, D = D, cov = cov)
    w = r_omega.gen(D = D, drop = drop_w)
    y, noise_data = g_lambda.gen(X = X, w = w, noise_level = noise)

    if drop_data != None:
        X_del, dp = data_drop(X=X, drop_rate=drop_data)
        filepath = save_dataset_avro(X=X_del, y=y, w=w, noise_data=noise_data, output_path=output_path, X_original=X, dp=dp)
    else:
        filepath = save_dataset_avro(X=X, y=y, w=w, noise_data=noise_data, output_path=output_path)

    return filepath, w
