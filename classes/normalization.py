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

        if self.per_line == False:
            self.x = scaler.fit_transform(self.x)
        else:
            for i in range(0, len(self.x)):
                xi = self.x[i].reshape(-1, 1)
                xi = scaler.fit_transform(xi)
                for j in range(0, len(xi)):
                    self.x[i][j] = xi[j]

        return self.x