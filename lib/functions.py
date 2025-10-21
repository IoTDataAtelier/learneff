import os
import numpy as np
import fastavro
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx


def save_dataset_avro(X, y, w, output_path, filename="synthetic_dataset.avro"):
    """
    Save dataset (X, y) into an Avro file and weights into a NumPy .npy file.

    Args:
        X (np.ndarray): Feature matrix of shape (N, D).
        y (np.ndarray): Target vector of shape (N, 1).
        w (np.ndarray): True weight vector of shape (D, 1).
        output_path (str): Directory where files will be saved.
        filename (str): Name of the Avro file (default: 'synthetic_dataset.avro').

    Returns:
        str: Path to the saved Avro file.
    """
    schema = {
        "type": "record",
        "name": "SyntheticData",
        "fields": [
            {"name": "features", "type": {"type": "array", "items": "double"}},
            {"name": "target", "type": "double"},
        ],
    }

    # Prepare records for Avro
    records = [
        {"features": X[i].tolist(), "target": float(y[i])}
        for i in range(len(y))
    ]

    # Save Avro dataset
    filepath = os.path.join(output_path, filename)
    with open(filepath, "wb") as out:
        fastavro.writer(out, schema, records)

    # Save true weights separately
    np.save(os.path.join(output_path, "true_weights.npy"), w)

    return filepath


def plot_graph(G, output_path, start_epoch, last_epoch):
    """
    Generic function to plot and save a NetworkX graph.

    Args:
        G (nx.Graph): The graph to plot.
        output_path (str): Directory where the plot will be saved.
        title (str): Title of the plot.
        filename (str): Output filename for the PNG image.
        seed (int): Random seed for layout reproducibility.

    Returns:
        str: Path to the saved graph image.
    """
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [abs(d['weight']) for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=weights, edge_cmap=cm.coolwarm)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Weight Correlation Graph (epochs {start_epoch}–{last_epoch})")
    plt.axis("off")
    plt.tight_layout()

    fname = os.path.join(output_path, f"graph_{start_epoch}_{last_epoch}.png")
    plt.savefig(fname)
    plt.close()


def plot_graph_heatmap(n_components: np.ndarray, output_path:str, T: int, S_w: int, M: int):
    fig, ax = plt.subplots()

    x_axis = list(range(0, T - S_w + 1, M))
    y_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    im = ax.imshow(n_components, origin="lower", cmap=cm.jet)
    #ax.set_xticks(x_axis)
    #ax.set_yticks(y_axis)
    #fig.colorbar(im, ax=ax, mappable=cm.ScalarMappable(cmap=cm.rainbow))

    ax.set_title("Number of Components")
    fig.tight_layout()
    
    fname = os.path.join(output_path, f"graph_heatmap.png")
    fig.savefig(fname)


def save_graph(G, output_path, start_epoch, last_epoch):
    fname = os.path.join(output_path, f"graph_{start_epoch}_{last_epoch}.gml")
    nx.write_gml(G, fname)

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]