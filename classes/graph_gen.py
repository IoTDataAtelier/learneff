from classes.base_class import BaseClass
from abc import abstractmethod

class GraphGenerationStrategy(BaseClass):

    @abstractmethod
    def gen(self, **kwargs):
        """
        Generating a set of graphs according to a set of parameters
        """
        pass

class Pairwise(GraphGenerationStrategy):

    def gen(self, **kwargs):
        self.set_attributes(kwargs)

        # Add nodes (weights/features)
        for i in range(self.D):
            self.G.add_node(i)

        # Add edges based on correlation
        for i in range(self.D):
            for j in range(i + 1, self.D):
                corr = self.corr_op(self.Wt[i], self.Wt[j])
                if abs(corr) > self.lim:
                    self.G.add_edge(i, j, weight=corr)

        return self.G