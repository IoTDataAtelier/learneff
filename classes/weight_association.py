from classes.base_class import BaseClass
from abc import abstractmethod

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

class Spearman(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)

class Kendall(AssociationStrategy):

    def eval(self, **kwargs):
        self.set_attributes(kwargs)