import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from classes.graph_gen import GraphGenerationStrategy
from classes.weight_association import AssociationStrategy
from classes.normalization import NormalizationStrategy

from lib.functions import plot_graph, save_graph

def remove_zero_weights(G: nx.Graph):
    for u, v in G.edges:
        if G[u][v]["weight"] == 0.0:
            G.remove_edge(u, v)
    return G

def normalize_weights(G: nx.Graph, norm_f: NormalizationStrategy):
    obtain_weights = lambda e : e[2]["weight"]
    edges = G.edges(data=True)
    weights = np.array(list(map(obtain_weights, edges)))
    weights = weights.reshape(-1, 1)
    weights = norm_f.norm(x = weights, per_line = False)
    weights = weights.flatten()

    i = 0
    for u, v in G.edges:
        G[u][v]["weight"] = weights[i]
        i += 1

    return G

def generate_graphs(W: np.ndarray, output_path: str, q: GraphGenerationStrategy, corr: AssociationStrategy, norm_f: NormalizationStrategy, S_w: int = 10, M: int = 5, min: float = -2):
    D, T = W.shape
    graphs = []

    for t in range(0, T - S_w + 1, M):
        e = t + S_w
        G = nx.Graph()

        G = q.gen(G = G, D = D, Wt = W[:, t:e], min = min, corr_op = corr)
        if norm_f != None:
            G = normalize_weights(G=G, norm_f=norm_f)

        G = remove_zero_weights(G)
        graphs.append(G)

        # --- Plot the graph ---
        plot_graph(G, output_path, t, e)

        # --- Save graph ---
        save_graph(G, output_path, t, e)

    return graphs
