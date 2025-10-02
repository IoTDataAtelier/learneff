from abc import ABC, abstractmethod
import numpy as np

class ErrorFunctionStrategy(ABC):

    @abstractmethod
    def eval(self, **kwargs):
        """
        Calculate the error based on a set of parameters
        """
        pass

class MeanSquaredError(ErrorFunctionStrategy):

    def eval(self, **kwargs):
        y_true = kwargs['y_true']
        y_pred = kwargs['y_pred']

        return np.mean((y_true - y_pred) ** 2)