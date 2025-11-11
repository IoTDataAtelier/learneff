from classes.base_class import BaseClass
from abc import abstractmethod
from sklearn.preprocessing import MinMaxScaler

class NormalizationStrategy(BaseClass):

    @abstractmethod
    def norm(self, **kwargs):
        """
        Normalize values of a random vector/matrix
        """
        pass
    
class MinMaxNorm(NormalizationStrategy):

    def norm(self, **kwargs):
        self.set_attributes(kwargs)

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.x = scaler.fit_transform(self.x)

        return self.x