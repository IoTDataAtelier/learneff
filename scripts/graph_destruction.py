import networkx as nx
import numpy as np
import os

def graph_edge_destruction(G: list, output_path:str):
    n_components = np.zeros((11, len(G)))
    W_sorted = np.zeros((len(G), G[0].number_of_edges(), 2))
    graph_n = 0
    obtain_weights = lambda e : e[2]["weight"]

    for g in G:
        sort_edges = sorted(g.edges(data=True), key = obtain_weights)
        n_edges = len(sort_edges)
        max_edge = obtain_weights(sort_edges[n_edges - 1])

        cont = 0
        n_components[0][graph_n] = nx.number_connected_components(g)
        for i in range(10, 101, 10):
            threshold = max_edge * (i/100)
            
            for j in range(cont, n_edges):
                W_sorted[graph_n, cont,:] = (obtain_weights(sort_edges[j]), nx.number_connected_components(g))
                if obtain_weights(sort_edges[j]) <= threshold:
                    u = sort_edges[j][0]
                    v = sort_edges[j][1]
                    g.remove_edge(u, v)
                    cont += 1
            
            n_components[int(i/10)][graph_n] = nx.number_connected_components(g)

        graph_n += 1

    np.save(os.path.join(output_path, "graph_destruction_components.npy"), n_components)
    
    return n_components, W_sorted

def edge_destruction(G:nx.graph, filter:np.ndarray, output_path:str, t:int, xt:list, yt:list):
    x = np.zeros(G.number_of_edges()) # Store the sorted edge weights
    y = np.zeros(G.number_of_edges()) # Store the number of components based o xi weight

    obtain_weights = lambda e : e[2]["weight"]
    sort_edges = sorted(G.edges(data=True), key = obtain_weights)
    
    n_edges = len(sort_edges)
    max_edge = obtain_weights(sort_edges[n_edges - 1])
    cont = 0

    for f in filter:
        threshold = max_edge * f

        for i in range(cont, n_edges):
            w = obtain_weights(sort_edges[i])
            if w <= threshold:
                u = sort_edges[i][0]
                v = sort_edges[i][1]
                G.remove_edge(u, v)

                x[i] = w
                y[i] = nx.number_connected_components(G)
                cont += 1
            else:
                break

    np.save(os.path.join(output_path, f"graph_{t}_weights.npy"), x)
    np.save(os.path.join(output_path, f"graph_{t}_components.npy"), y)

    xt.append(x)
    yt.append(y)

    return xt, yt