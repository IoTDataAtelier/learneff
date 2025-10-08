from abc import ABC, abstractmethod
import numpy as np

class AlgorithmStrategy(ABC):

    @abstractmethod
    def update(self, **kwargs):
        """
        Generating new values for w
        """
        pass

class NewtonPLA(AlgorithmStrategy):

    def update(self, **kwargs):

        X = kwargs['X']
        y = kwargs['y']
        w = kwargs['w']
        learning_rate = kwargs['lr']

        wt = (1 - learning_rate) * w + learning_rate * np.linalg.inv(np.dot(X.T, X)) @ np.dot(X.T, y)
        return wt