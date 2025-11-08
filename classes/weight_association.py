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
    
class ICC(AssociationStrategy):

    def __init__(self, q = 1):
        self.q = q
        self.dim = {}

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

        if self.dx not in self.dim:
            self.dim[self.dx] = self.sq(self.x)

        if self.dy not in self.dim:
            self.dim[self.dy] = self.sq(self.y)
            
        mxy = self.mq(self.dim[self.dx], self.dim[self.dy])
        
        values, quant = np.unique(mxy, return_counts=True)
        p = np.array([i/len(values) for i in quant])
        h = st.entropy(p, base = 2)

        if h == 0:
            return 0
        else:
            if len(values) == 2:
                p1 = p[0]
                p2 = p[1]
            if values % 2 == 0:
                half = (self.q + 1)/2
                p1 = sum(p[0: half])
                p2 = sum(p[half: self])
            else:
                t = np.median(values)
                p1 = sum(p[values > t])
                p2 = sum(p[values < t])

            if p1 > p2:
                icc = 1 - h
            else:
                icc = h - 1

            return icc

    def mq(self, sx, sy):
        mxy = []
        for i in range(0, len(sx), self.q):
            j = i + self.q
            m = sx[i:j] ^ sy[i:j]
            mxy.append(m)
        
        return np.array(mxy)

    def sq(self, v: np.ndarray):
        sqv = []
        for i in range(self.q, len(v)):
            for j in range(i - self.q, i):
                if v[i] >= v[j]:
                    s = 1
                else:
                    s = 0
                sqv.append(s)

        return np.array(sqv)