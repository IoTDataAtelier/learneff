from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np

class AlgorithmStrategy(BaseClass):

    @abstractmethod
    def update(self, **kwargs):
        """
        Generating new values for w
        """
        pass

class Newton(AlgorithmStrategy):

    def update(self, **kwargs):
        self.set_attributes(kwargs)

        wt = (1 - self.lr) * self.w + self.lr * np.linalg.inv(np.dot(self.X.T, self.X)) @ np.dot(self.X.T, self.y)
        return wt
    
class SteepestDescent(AlgorithmStrategy):

    def update(self, **kwargs):
        self.set_attributes(kwargs)

        grad = -(2 / len(self.X)) * self.X.T @ (self.y - self.y_pred)
        self.w -= self.lr * grad
        return self.w