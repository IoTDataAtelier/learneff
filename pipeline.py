from datetime import datetime
import os
import numpy as np

#---- Classes ----
from classes.data_generation import MultivariateGaussian, RandomColumnVector, LinearPlusNoise, RandomRowVector
from classes.error_function import MeanSquaredError
from classes.model import Linear
from classes.algorithm import Newton, GradientDescent
from classes.graph_gen import Pairwise
from classes.weight_association import Pearson, Spearman, Kendall
# -----------------------

from pipeline_builder import PipelineBuilder

def run_scene(pipeline: PipelineBuilder, scene: int):
    
    # ---- Variable config ----
    D = 11           # number of features
    N = 100          # number of samples
    T = 100          # number of epochs
    LR = 0.001        # learning rate
    NOISE = 1.0      # noise level
    S_W = 10         # sliding window size for graphs
    M = 5            # stride between windows
    COV = np.eye(D-1)
    # -----------------------

    output_path = f"output/experiment/scene_{scene}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/graph", exist_ok=True)
    os.makedirs(f"{output_path}/plot/AUC", exist_ok=True)

    pipeline.data_generation(output_path=output_path, f_theta=MultivariateGaussian(), r_omega=RandomColumnVector(), g_lambda=LinearPlusNoise(), N=N, D=D, noise=NOISE, cov=COV)
    pipeline.model_training(output_path=output_path, D=D, T=T, lr=LR, r_omega=RandomColumnVector(), e_phi=MeanSquaredError(), H = Linear(), a = GradientDescent())                
    
    corr_weights = {"pearson": Pearson(), "spearman": Spearman(), "kendall": Kendall()}
    for n, c in corr_weights.item():
        pipeline.state.update({f"graphs_{n}": None, f"n_components_{n}": None, f"W_sorted_{n}": None})

        pipeline.graph_generation(output_path=f"{output_path}/graph", q=Pairwise(), corr=c, S_w=S_W, M=M)
        pipeline.graph_destruction(output_path=output_path)
        pipeline.plot_destruction_heatmap(output_path=f"{output_path}/plot", T=T, S_w=S_W, M=M)
        pipeline.plot_destruction_AUC(output_path=f"{output_path}/plot/AUC", time_windows=list(range(0, T - S_W + 1, M)))
        # CDF


def run_all(pipeline: PipelineBuilder):

    # ---- Variable config ----
    D = 10           # number of features
    N = 100          # number of samples
    T = 100          # number of epochs
    LR = 0.01        # learning rate
    NOISE = 0.7      # noise level
    S_W = 10         # sliding window size for graphs
    M = 5            # stride between windows
    COV = np.eye(D-1)
    # -----------------------

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"experiment_{timestamp}"
    output_path = f"output/{experiment_dir}"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/graph", exist_ok=True)

    pipeline.data_generation(output_path=output_path, f_theta=MultivariateGaussian(), r_omega=RandomColumnVector(), g_lambda=LinearPlusNoise(), N=N, D=D, noise=NOISE, cov=COV)
    pipeline.model_training(output_path=output_path, D=D, T=T, lr=LR, r_omega=RandomColumnVector(), e_phi=MeanSquaredError(), H = Linear(), a = GradientDescent())                
    pipeline.graph_generation(output_path=f"{output_path}/graph", q=Pairwise(), corr=Kendall(), S_w=S_W, M=M)
    pipeline.graph_destruction(output_path=output_path)
    pipeline.plot_destruction_heatmap(output_path=output_path, T=T, S_w=S_W, M=M)
    pipeline.plot_destruction_AUC(output_path=output_path, time_windows=list(range(0, T - S_W + 1, M)))

def run_pipeline():
    state = {"filepath": "", "w_true": None, "W": None, "graphs": None, "n_components": None, "W_sorted": None}

    pipeline = PipelineBuilder(state)
    
    # run_all(pipeline)
    for i in range(1, 4):
        run_scene(pipeline, i)
        pipeline.execute_pipeline()

if __name__ == "__main__":
    run_pipeline()
