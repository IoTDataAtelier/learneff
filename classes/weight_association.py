from classes.base_class import BaseClass
from abc import abstractmethod
import numpy as np
import scipy.stats as st
import scipy.signal as ss
from numpy.linalg import norm

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

        sr = st.spearmanr(self.x, self.y, axis=1)
        return sr.statistic

class Kendall(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        sr = st.kendalltau(self.x, self.y)
        return sr.statistic
    
class CrossCorrelation(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        cc = ss.correlate(self.x, self.y)
        return cc
    
class Cosine(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        c = np.dot(self.x, self.y) / (norm(self.x) * norm(self.y))
        return c