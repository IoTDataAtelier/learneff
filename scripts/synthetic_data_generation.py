import numpy as np
from lib.functions import save_dataset_avro
from classes.data_generation import DataGenerationStrategy
import scipy.stats as st

#def f_theta(N: int, D: int) -> np.ndarray:
#    return np.random.randn(N, D)


#def r_omega(D: int) -> np.ndarray:
    #return np.random.randn(D, 1)
#    v = np.array(st.norm.rvs(size = D)).T
#    return v.reshape(-1, 1)

#def g_lambda(X: np.ndarray, w: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
#    noise = noise_level * np.random.randn(X.shape[0], 1)
#    return X @ w + noise


def synthetic_data_generation(output_path: str, 
                              f_theta: DataGenerationStrategy, 
                              r_omega: DataGenerationStrategy, 
                              g_lambda: DataGenerationStrategy,
                              N: int = 200, D: int = 5, noise: float = 0.1
                              ):
    X = f_theta.gen(N = N, D = D)
    w = r_omega.gen(D = D)
    y = g_lambda.gen(X = X, w = w, noise_level = noise)

    filepath = save_dataset_avro(X, y, w, output_path)

    return filepath, w
