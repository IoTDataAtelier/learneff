from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np

class ErrorFunctionStrategy(BaseClass):

    @abstractmethod
    def eval(self, **kwargs):
        """
        Calculate the error based on a set of parameters
        """
        pass

class MeanSquaredError(ErrorFunctionStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        return np.mean((self.y_true - self.y_pred) ** 2)