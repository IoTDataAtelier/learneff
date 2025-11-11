from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np

class NormalizationStrategy(BaseClass):

    @abstractmethod
    def update(self, **kwargs):
        """
        Normalize values of a random vector/matrix
        """
        pass

class SumNorm(NormalizationStrategy):

    def update(self, **kwargs):
        self.set_attributes(kwargs)

        s = sum(self.x)
        m = lambda v : v/s

        if len(self.x.shape) == 2:
            for i in range(0, self.x.shape[0]):
                for j in range(0, self.x.shape[0]):
                    self.x[i] = m(self.x[i])
        else: # shape 1
            for i in range(0, len(self.x) - 1):
                self.x[i] = m(self.x[i])