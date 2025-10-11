from abc import ABC, abstractmethod
import numpy as np

class AlgorithmStrategy(ABC):

    @abstractmethod
    def update(self, **kwargs):
        """
        Generating new values for w
        """
        pass

    def set_attributes(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)

class NewtonPLA(AlgorithmStrategy):

    def update(self, **kwargs):
        self.set_attributes(kwargs)

        wt = (1 - self.lr) * self.w + self.lr * np.linalg.inv(np.dot(self.X.T, self.X)) @ np.dot(self.X.T, self.y)
        return wt