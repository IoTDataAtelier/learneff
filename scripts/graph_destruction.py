import networkx as nx

def graph_edge_destruction(G: list):

    for g in G:
        sort_edges = sorted(g.edges(data=True), key = lambda e : e[2]["weight"])
        n_edges = len(sort_edges)
        max_edge = sort_edges[n_edges - 1][2]["weight"]

        cont = 0
        n_components = []
        for i in range(10, 101, 10):
            threshold = max_edge * (i/100)
            
            for j in range(cont, n_edges):
                if sort_edges[j][2]["weight"] < threshold:
                    u = sort_edges[j][0]
                    v = sort_edges[j][1]
                    g.remove_edge(u, v)
                else:
                    cont = j

            

G = [nx.Graph()]
G[0].add_node(1)
G[0].add_node(2)
G[0].add_node(3)

G[0].add_edge(1, 2, weight=0.2)
G[0].add_edge(1, 3, weight=0.4)

nx.components(G)

graph_edge_destruction(G)