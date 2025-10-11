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

    def set_attributes(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)

class MultivariateGaussian(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        mean = np.array(st.norm.rvs(size = self.D)).T
        cov = np.eye(self.D)      

        mn = st.multivariate_normal(mean = mean, cov = cov, seed = 1)
        M = mn.rvs(self.N)
        #ones = np.ones((N, 1))
        #X = np.column_stack([ones, M])
        return M
    
class LegendrePolynomials(DataGenerationStrategy):

    def gen(self, **kwargs):
        pass

class RandomColumnVector(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)
        
        w = np.array(st.norm.rvs(size = self.D)).T
        w = w.reshape(-1, 1)
        return w
    
class LinearPlusNoise(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        noise = self.noise_level * np.random.randn(self.X.shape[0], 1)
        return self.X @ self.w + noise