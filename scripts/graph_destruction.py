import networkx as nx
import numpy as np
import os

def graph_edge_destruction(G: list, output_path:str):
    n_components = np.zeros((11, len(G)))
    W_sorted = np.zeros(len(G))
    graph_n = 0
    obtain_weights = lambda e : e[2]["weight"]

    for g in G:
        sort_edges = sorted(g.edges(data=True), key = obtain_weights)
        W_sorted[graph_n] = list(map(obtain_weights, sort_edges))
        n_edges = len(sort_edges)
        max_edge = obtain_weights(sort_edges[n_edges - 1])

        cont = 0
        n_components[0][graph_n] = nx.number_connected_components(g)
        for i in range(10, 101, 10):
            threshold = max_edge * (i/100)
            
            for j in range(cont, n_edges):
                if obtain_weights(sort_edges[j]) <= threshold:
                    u = sort_edges[j][0]
                    v = sort_edges[j][1]
                    g.remove_edge(u, v)
                    cont += 1
            
            n_components[int(i/10)][graph_n] = nx.number_connected_components(g)

        graph_n += 1

    np.save(os.path.join(output_path, "graph_destruction_components.npy"), n_components)
    
    return n_components, W_sorted