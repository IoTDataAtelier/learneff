import os
import numpy as np
import fastavro
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    nx.draw_networkx_edges(G, pos, width=weights, edge_color=weights, edge_cmap=cm.coolwarm, edge_vmin=0, edge_vmax=1)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Weight Correlation Graph (epochs {start_epoch}–{last_epoch})")
    plt.axis("off")
    plt.tight_layout()

    fname = os.path.join(output_path, f"graph_{start_epoch}_{last_epoch}.png")
    plt.savefig(fname)
    plt.close()


def plot_graph_destruction_heatmap(n_components: np.ndarray, output_path:str, T: int, S_w: int, M: int):
    x_axis = list(range(0, T + 1, 10))
    y_axis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    fig, ax = plt.subplots(figsize=(5.7, 5.7))

    # ---- Adjust colobar size ----
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)

    # ---- Plot heatmap ----
    im = ax.imshow(n_components, origin="lower", cmap=cm.Spectral_r, aspect='auto', interpolation="nearest")

    x_ticks = np.linspace(0, n_components.shape[1] - 1, len(x_axis))
    y_ticks = np.linspace(0, n_components.shape[0] - 1, len(y_axis))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}" for x in x_axis])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y}" for y in y_axis])

    ax.set_xlabel("Time Window")
    ax.set_ylabel("Filter")

    fig.colorbar(im, cax=ax_cb)

    ax.set_title("Number of Components")
    fig.tight_layout()
    
    fname = os.path.join(output_path, f"graph_heatmap.png")
    fig.savefig(fname)
    plt.close()

def plot_AUC(t: int, Wt: np.ndarray, AUC: float, output_path:str):
    fig, ax = plt.subplots()

    n_weights = len(Wt)
    x = np.zeros(n_weights)
    y = np.zeros(n_weights)
    
    for i in range(0, n_weights):
        x[i] = Wt[i][0]
        y[i] = Wt[i][1]
    
    ax.plot(x, y, color='green', marker='o', linestyle='solid')

    x_axis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_bins = 5

    x_ticks = np.linspace(0, max(x), len(x_axis))
    y_ticks = np.linspace(1, max(y), y_bins)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}" for x in x_axis])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    ax.set_xlabel("Edge Weight")
    ax.set_ylabel("Number of Components")

    ax.set_title(f"Time Window = {t}, AUC = {AUC:.4f}")
    fig.tight_layout()

    fname = os.path.join(output_path, f"graph_AUC_{t}.png")
    fig.savefig(fname)
    plt.close()

def save_graph(G, output_path, start_epoch, last_epoch):
    fname = os.path.join(output_path, f"graph_{start_epoch}_{last_epoch}.gml")
    nx.write_gml(G, fname)