import networkx as nx
import numpy as np
from lib.functions import plot_graph_heatmap

def graph_edge_destruction(G: list, output_path:str, T: int, S_w: int, M: int):
    n_components = np.zeros((len(G), 11))
    graph_n = 0

    for g in G:
        sort_edges = sorted(g.edges(data=True), key = lambda e : e[2]["weight"])
        n_edges = len(sort_edges)
        max_edge = sort_edges[n_edges - 1][2]["weight"]

        cont = 0
        n_components[graph_n][0] = nx.number_connected_components(g)
        for i in range(10, 101, 10):
            threshold = max_edge * (i/100)
            
            for j in range(cont, n_edges):
                if sort_edges[j][2]["weight"] < threshold:
                    u = sort_edges[j][0]
                    v = sort_edges[j][1]
                    g.remove_edge(u, v)
                else:
                    cont = j
            
            n_components[graph_n][i] = nx.number_connected_components(g)

        graph_n += 1
            
        plot_graph_heatmap(n_components, output_path, T, S_w, M)

G = [nx.Graph()]
G[0].add_node(1)
G[0].add_node(2)
G[0].add_node(3)

G[0].add_edge(1, 2, weight=0.2)
G[0].add_edge(1, 3, weight=0.4)

nx.components(G)

graph_edge_destruction(G)