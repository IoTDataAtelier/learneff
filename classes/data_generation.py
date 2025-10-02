from abc import ABC, abstractmethod
import scipy.stats as st
import numpy as np

class DataGenerationStrategy(ABC):

    @abstractmethod
    def gen(self, **kwargs):
        """
        Generating data according to a set of parameters
        """
        pass

class MultivariateGaussian(DataGenerationStrategy):

    def gen(self, **kwargs):
        D = kwargs['D']
        N = kwargs['N']

        mean = np.array(st.norm.rvs(size = D)).T
        cov = np.eye(D)      

        mn = st.multivariate_normal(mean = mean, cov = cov, seed = 1)
        M = mn.rvs(N)
        #ones = np.ones((N, 1))
        #X = np.column_stack([ones, M])
        return M
    
class LegendrePolynomials(DataGenerationStrategy):

    def gen(self, **kwargs):
        pass

class RandomColumnVector(DataGenerationStrategy):

    def gen(self, **kwargs):
        D = kwargs['D']
        
        w = np.array(st.norm.rvs(size = D)).T
        w = w.reshape(-1, 1)
        return w
    
class LinearPlusNoise(DataGenerationStrategy):

    def gen(self, **kwargs):
        X = kwargs['X']
        w = kwargs['w']
        noise_level = kwargs['noise_level']

        noise = noise_level * np.random.randn(X.shape[0], 1)
        return X @ w + noise