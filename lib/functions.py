import os
import numpy as np
import fastavro
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import scipy.stats as st


def save_dataset_avro(X, y, w, output_path, filename="synthetic_dataset.avro", X_original = None, dp = None):
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

    if dp != None:
        schema["fields"].append({"name": "features_without_drop", "type": {"type": "array", "items": "double"}})
        
        for i in range(len(y)):
            records[i]["features_without_drop"] = X_original[i].tolist()
            
        np.save(os.path.join(output_path, "dropped_columns.npy"), dp)

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

def plot_AUC(t: int, x, y, AUC: float, output_path:str):
    fig, ax = plt.subplots()
    
    ax.plot(x, y, marker='o', markeredgecolor='black', markeredgewidth=1, linestyle='solid')

    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=1)
    ax.set_xlabel("Edge Weight")
    ax.set_ylabel("Number of Components")

    ax.set_title(f"Time Window = {t}, AUC = {AUC:.4f}")
    fig.tight_layout()

    fname = os.path.join(output_path, f"graph_AUC_{t}.png")
    fig.savefig(fname)
    plt.close()

def plot_error_train_val(partial_filepath: str, scenes: list, T: int):

    fig, ax = plt.subplots(figsize = (8, 4))
    ax.set_ylabel("Error")
    ax.set_xlabel("Epochs")

    epochs = range(1, T + 1)
    colors = cm._colormaps['tab10'].colors[:len(scenes)]

    i = 0
    for s in scenes:
        train_error = np.load(os.path.join(partial_filepath, f"scene_{s}/train_errors.npy"))
        val_error = np.load(os.path.join(partial_filepath, f"scene_{s}/validation_errors.npy"))

        ax.plot(epochs, train_error, label = f"train_sc{s}", color=colors[i], marker='o', markevery=5)
        ax.plot(epochs, val_error, label = f"val_sc{s}", color=colors[i], marker='D', markevery=5)

        i += 1

    x_ticks = range(0, T + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}" for x in x_ticks])
    
    #ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=T)

    ax.set_title("Train and Validation Error")

    ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
    fig.tight_layout()

    fname = os.path.join(partial_filepath, f"errors.png")
    fig.savefig(fname)
    plt.close()

def plot_weight_CDF(G: list, output_path:str, time_windows: list):

    for i in range(0, len(G)):
        fig, ax = plt.subplots()

        graph = G[i]
        obtain_weights = lambda e : e[2]["weight"]
        edges = graph.edges(data=True)
        weights = np.array(list(map(obtain_weights, edges)))

        mean = np.mean(weights)
        std = np.std(weights)
        dx = 1e-5

        # Generate Bins
        a = mean - 3 * std
        b = mean + 3 * std
        n = round((b - a)/dx)
        xs = np.linspace(a, b, n)
        
        ax.plot(xs, st.norm.cdf(xs, loc=mean, scale=std))

        ax.axvline(mean, color='k', linestyle='dashed', linewidth=2)
        ax.set_xlabel("weights")
        ax.set_ylabel("P(weights <= x)")
        plt.title(f"Graph Weight's CDF, Time Window = {time_windows[i]}")

        fname = os.path.join(output_path, f"cdf_{time_windows[i]}.png")
        fig.savefig(fname)
        plt.close()


def save_graph(G, output_path, start_epoch, last_epoch):
    fname = os.path.join(output_path, f"graph_{start_epoch}_{last_epoch}.gml")
    nx.write_gml(G, fname)