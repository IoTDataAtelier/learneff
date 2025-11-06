from classes.base_class import BaseClass
from abc import abstractmethod
import scipy.stats as st
import numpy as np
import random

class DataGenerationStrategy(BaseClass):

    @abstractmethod
    def gen(self, **kwargs):
        """
        Generating data according to a set of parameters
        """
        pass

class MultivariateGaussian(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        mean = np.array(st.norm.rvs(size = self.D - 1)).T
        #cov = np.eye(self.D - 1)      

        mn = st.multivariate_normal(mean = mean, cov = self.cov, seed = 1)
        M = mn.rvs(self.N)
        ones = np.ones((self.N, 1))
        X = np.column_stack([ones, M])
        return X
    
class LegendrePolynomials(DataGenerationStrategy):

    def gen(self, **kwargs):
        pass

class RandomRowVector(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        return np.random.randn(self.N, self.D)

class RandomColumnVector(DataGenerationStrategy):

    def gen(self, drop = None, **kwargs):
        self.set_attributes(kwargs)
        
        w = np.array(st.norm.rvs(size = self.D)).T
        w = w.reshape(-1, 1)

        # Drop is a number [0, 1) repressenting how many numbers to take off
        if drop != None:
            dp = random.sample(range(1, self.D), int((self.D - 1) * drop))
            for i in dp:
                w[i] = 0

        return w
    
class LinearPlusNoise(DataGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        noise = self.noise_level * np.random.randn(self.X.shape[0], 1)
        return self.X @ self.w + noise