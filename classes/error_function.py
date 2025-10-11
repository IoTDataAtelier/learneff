from abc import ABC, abstractmethod
import numpy as np

class ErrorFunctionStrategy(ABC):

    @abstractmethod
    def eval(self, **kwargs):
        """
        Calculate the error based on a set of parameters
        """
        pass

    def set_attributes(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)

class MeanSquaredError(ErrorFunctionStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        return np.mean((self.y_true - self.y_pred) ** 2)