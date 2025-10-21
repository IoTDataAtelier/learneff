import networkx as nx
import numpy as np
from lib.functions import plot_graph_heatmap

def graph_edge_destruction(G: list, output_path:str, T: int, S_w: int, M: int):
    n_components = np.zeros((11, len(G)))
    graph_n = 0

    for g in G:
        sort_edges = sorted(g.edges(data=True), key = lambda e : e[2]["weight"])
        n_edges = len(sort_edges)
        max_edge = sort_edges[n_edges - 1][2]["weight"]

        cont = 0
        n_components[0][graph_n] = nx.number_connected_components(g)
        for i in range(10, 101, 10):
            threshold = max_edge * (i/100)
            
            for j in range(cont, n_edges):
                if sort_edges[j][2]["weight"] <= threshold:
                    u = sort_edges[j][0]
                    v = sort_edges[j][1]
                    g.remove_edge(u, v)
                    cont += 1
            
            n_components[int(i/10)][graph_n] = nx.number_connected_components(g)

        graph_n += 1
            
        plot_graph_heatmap(n_components, output_path, T, S_w, M)
