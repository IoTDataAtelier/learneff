from abc import ABC, abstractmethod
import numpy as np

class ModelStrategy(ABC):

    @abstractmethod
    def pred(self, **kwargs):
        """
        Model function prediction for a set of parameters
        """
        pass

    def set_attributes(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)

class Linear(ModelStrategy):

    def pred(self, **kwargs):
        self.set_attributes(kwargs)

        return np.dot(self.X, self.w)
