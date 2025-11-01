import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from classes.graph_gen import GraphGenerationStrategy
from classes.weight_association import AssociationStrategy

from lib.functions import plot_graph, save_graph


#def q(G: nx.graph, D: int, W: np.ndarray, t: int, e: int):
    # Add nodes (weights/features)
#    for i in range(D):
#        G.add_node(i)

    # Add edges based on correlation
#    for i in range(D):
#        for j in range(i + 1, D):
#            corr = pearson_corr(W[i, t:e], W[j, t:e])
#            if abs(corr) > 0.5:
#                G.add_edge(i, j, weight=corr)
#    return G

def generate_graphs(W: np.ndarray, output_path: str, q: GraphGenerationStrategy, corr: AssociationStrategy, S_w: int = 10, M: int = 5, min: float = -2):
    D, T = W.shape
    graphs = []

    for t in range(0, T - S_w + 1, M):
        e = t + S_w
        G = nx.Graph()

        G = q.gen(G = G, D = D, Wt = W[:, t:e], min = min, corr_op = corr)

        graphs.append(G)

        # --- Plot the graph ---
        plot_graph(G, output_path, t, e)

        # --- Save graph ---
        save_graph(G, output_path, t, e)

    return graphs
