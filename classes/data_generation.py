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

        mean = st.norm.rvs(D)
        cov = np.eye(D)      

        print(cov)

        mn = st.multivariate_normal(mean, cov, seed = 1)
        M = mn.rvs(N)
        #ones = np.ones((N, 1))
        #X = np.column_stack([ones, M])
        return M