from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np

class ModelStrategy(BaseClass):

    @abstractmethod
    def pred(self, **kwargs):
        """
        Model function prediction for a set of parameters
        """
        pass

class Linear(ModelStrategy):

    def pred(self, **kwargs):
        self.set_attributes(kwargs)

        return np.dot(self.X, self.w)
