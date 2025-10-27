from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np
import scipy.stats as st

class AssociationStrategy(BaseClass):

    @abstractmethod
    def eval(self, **kwargs):
        """
        Evaluates the association between parameters
        """
        pass

class Pearson(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        if np.std(self.x) == 0 or np.std(self.y) == 0:
            return 0.0
        
        return np.corrcoef(self.x, self.y)[0, 1]

class Spearman(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)



class Kendall(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)