from datetime import datetime
import os
from lib.logger import success
from lib.functions import pearson_corr, plot_graph_destruction_heatmap
#from scipy.stats import pearsonr

#---- Experiment Phases ----
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from scripts.graph_destruction import graph_edge_destruction
from scripts.components_AUC import graph_components_AUC
# -----------------------

#---- Classes ----
from classes.data_generation import MultivariateGaussian, RandomColumnVector, LinearPlusNoise
from classes.error_function import MeanSquaredError
from classes.model import Linear
from classes.algorithm import NewtonPLA
from classes.graph_gen import Pairwise
# -----------------------

# ---- Global config ----
D = 10           # number of features
N = 100          # number of samples
T = 100          # number of epochs
LR = 0.01        # learning rate
NOISE = 0.7      # noise level
S_W = 10         # sliding window size for graphs
M = 5            # stride between windows
# -----------------------


def run_pipeline():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"experiment_{timestamp}"
    output_path = f"output/{experiment_dir}"
    os.makedirs(output_path, exist_ok=True)

    state = {"filepath": "", "w_true": None, "W": None, "graphs": None, "n_components": None, "W_sorted": None}

    pipeline_steps = [
        (
            "Generating Synthetic Data",
            lambda: (
                state.update(dict(zip(
                    ["filepath", "w_true"],
                    synthetic_data_generation(output_path, f_theta=MultivariateGaussian(), r_omega=RandomColumnVector(), g_lambda=LinearPlusNoise(), N=N, D=D, noise=NOISE)))
                )
            ),
        ),
        (
            "Training Process",
            lambda: state.update(
                W=training_process(
                    output_path, filepath=state["filepath"], D=D, T=T, lr=LR, r_omega=RandomColumnVector(), e_phi=MeanSquaredError(), H = Linear(), a = NewtonPLA()
                )
            ),
        ),
        (
            "Graph Generation and Graph Plot",
            lambda: state.update(
                graphs=generate_graphs(state["W"], output_path, q = Pairwise(), corr = pearson_corr, S_w=S_W, M=M
                )
            ),
        ),
        (
            "Graph Edge Destruction",
            lambda: state.update(dict(zip(
                ["n_components", "W_sorted"], 
                graph_edge_destruction(G=state["graphs"], output_path=output_path))
                )
            ),
        ),
        (
            "Plot Graph Destruction Heatmap",
            lambda: plot_graph_destruction_heatmap(
                n_components=state["n_components"], output_path=output_path, T=T, S_w=S_W, M=M
            )
        ),
        (
            "Plot Graph Components + AUC",
            lambda: graph_components_AUC(
                W_sorted=state["W_sorted"], time_windows=list(range(0, T - S_W + 1, M)), output_path=output_path
            )
        )
    ]

    for index, (step_name, step_action) in enumerate(pipeline_steps):
        success(f"Executing step {index + 1}: {step_name}\n")
        step_action()

    success(f"\nPipeline execution completed successfully. Results stored in: {output_path}")


if __name__ == "__main__":
    run_pipeline()
