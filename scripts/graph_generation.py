import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from lib.functions import plot_graph


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def q(G: nx.graph, D: int, W: np.ndarray, t: int, e: int):
    # Add nodes (weights/features)
    for i in range(D):
        G.add_node(i)

    # Add edges based on correlation
    for i in range(D):
        for j in range(i + 1, D):
            corr = pearson_corr(W[i, t:e], W[j, t:e])
            if abs(corr) > 0.5:
                G.add_edge(i, j, weight=corr)

    return G

def generate_graphs(W: np.ndarray, output_path: str, S_w: int = 10, M: int = 5):
    D, E = W.shape
    graphs = []

    for t in range(0, E - S_w + 1, M):
        e = t + S_w
        G = nx.Graph()

        G = q(G, D, W)

        graphs.append(G)

        # --- Plot the graph ---
        plot_graph(G, output_path, t, e)

    return graphs
