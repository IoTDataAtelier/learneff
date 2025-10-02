from abc import ABC, abstractmethod
import numpy as np

class ModelStrategy(ABC):

    @abstractmethod
    def pred(self, **kwargs):
        """
        Model function prediction for a set of parameters
        """
        pass

class Linear(ModelStrategy):

    def pred(self, **kwargs):
        X = kwargs['X']
        w = kwargs['w']

        return np.dot(X, w)
