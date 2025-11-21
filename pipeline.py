from datetime import datetime
import os
import numpy as np

#---- Classes ----
from classes.data_generation import MultivariateGaussian, RandomColumnVector, LinearPlusNoise, RandomRowVector
from classes.error_function import MeanSquaredError
from classes.model import Linear
from classes.algorithm import Newton, GradientDescent
from classes.graph_gen import Pairwise
from classes.weight_association import Pearson, CrossCorrelation, Cosine, ICC
from classes.normalization import MinMaxNorm
# -----------------------

from pipeline_builder import PipelineBuilder

def run_scene(pipeline: PipelineBuilder, scene: int, initial_path: str, D: int, drop_w = None, drop_data = None):
    
    # ---- Variable config ----
    N = 100          # number of samples
    T = 100          # number of epochs
    LR = 0.001        # learning rate
    NOISE = 20     # noise level
    S_W = 5         # sliding window size for graphs
    M = 2            # stride between windows
    COV = np.eye(D-1)
    time_windows = list(range(0, T - S_W + 1, M))
    filter = np.arange(0.0, 1.01, 0.1)
    # -----------------------

    output_path = f"{initial_path}/scene_{scene}"
    os.makedirs(output_path, exist_ok=True)

    pipeline.data_generation(output_path=output_path, f_theta=MultivariateGaussian(), r_omega=RandomColumnVector(), g_lambda=LinearPlusNoise(), N=N, D=D, noise=NOISE, cov=COV, drop_w=drop_w, drop_data=drop_data)
    
    if drop_data != None:
        D = int((D - 1) * drop_data) + 1
    
    pipeline.model_training(output_path=output_path, D=D, T=T, lr=LR, r_omega=RandomColumnVector(), e_phi=MeanSquaredError(), H = Linear(), a = GradientDescent())                
    pipeline.normalize_data(norm_f = MinMaxNorm(), norm_state = "W")
    pipeline.plot_train_val(partial_filepath=initial_path, scenes=[scene], T=T, output_path=output_path, val=True, train=True, filename=f"scene_{scene}_errors")

    corr_weights = {"pearson": Pearson(), "cross_correlation": CrossCorrelation(), "cosine": Cosine(), "icc": ICC()}
    #corr_weights = {"icc": ICC()}

    for n, c in corr_weights.items():
        graphs_state = f"graphs_{n}"
        x_state = f"x_{n}"
        y_state = f"y_{n}"
        AUC_state = f"AUC_{n}"

        pipeline.state.update({f"{graphs_state}": None, f"{x_state}": [], f"{y_state}": [], f"{AUC_state}": []})

        corr_output = f"{output_path}/{n}/fixed_window"
        graphs_output = f"{corr_output}/graph"
        data_output = f"{corr_output}/data"
        destruction_output = f"{data_output}/destruction_data"
        AUC_data_output = f"{data_output}/AUC"
        plots_output = f"{corr_output}/plots"
        AUC_output = f"{plots_output}/AUC"
        CDF_output = f"{plots_output}/CDF"

        os.makedirs(graphs_output, exist_ok=True)
        os.makedirs(AUC_output, exist_ok=True)
        os.makedirs(CDF_output, exist_ok=True)
        os.makedirs(data_output, exist_ok=True)
        os.makedirs(destruction_output, exist_ok=True)
        os.makedirs(AUC_data_output, exist_ok=True)

        if n == "cross_correlation":
            norm = MinMaxNorm()
        else:
            norm = None

        pipeline.graph_generation(q=Pairwise(), corr=c, S_w=S_W, M=M, graphs_state=graphs_state, output_path=graphs_output, norm_f=norm)
        pipeline.plot_CDF(graphs_state=graphs_state, time_windows=time_windows, output_path=CDF_output)
        
        for i in range(0, len(time_windows)):
            pipeline.graph_destruction(graphs_state=graphs_state, filter=filter, i=i, t=time_windows[i], x_state=x_state, y_state=y_state, output_path=destruction_output)
            
        pipeline.normalize_data(norm_f=MinMaxNorm(), norm_state=y_state, per_line=True)

        for i in range(0, len(time_windows)):
            t = time_windows[i]
            pipeline.calculate_AUC(t=t, output_path=AUC_data_output, x_state=x_state, y_state=y_state, AUC_state=AUC_state, i=i)
            pipeline.plot_AUC(time_window=[t], x_label="Normalized Edge Weight", y_label="Normalized Number of Components", analysis_type="Time Window, Iteration", output_path=AUC_output, AUC_data_output=AUC_data_output)
        
        pipeline.plot_destruction_heatmap(time_windows=time_windows, x_label="Iterations", AUC_data_output=AUC_data_output, output_path=plots_output)


# def run_all(pipeline: PipelineBuilder):

#     # ---- Variable config ----
#     D = 10           # number of features
#     N = 100          # number of samples
#     T = 100          # number of epochs
#     LR = 0.01        # learning rate
#     NOISE = 0.7      # noise level
#     S_W = 10         # sliding window size for graphs
#     M = 5            # stride between windows
#     COV = np.eye(D-1)
#     # -----------------------

#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     experiment_dir = f"experiment_{timestamp}"
#     output_path = f"output/{experiment_dir}"
#     os.makedirs(output_path, exist_ok=True)
#     os.makedirs(f"{output_path}/graph", exist_ok=True)

#     pipeline.data_generation(output_path=output_path, f_theta=MultivariateGaussian(), r_omega=RandomColumnVector(), g_lambda=LinearPlusNoise(), N=N, D=D, noise=NOISE, cov=COV)
#     pipeline.model_training(output_path=output_path, D=D, T=T, lr=LR, r_omega=RandomColumnVector(), e_phi=MeanSquaredError(), H = Linear(), a = GradientDescent())                
#     pipeline.graph_generation(output_path=f"{output_path}/graph", q=Pairwise(), corr=Kendall(), S_w=S_W, M=M)
#     pipeline.graph_destruction(output_path=output_path)
#     pipeline.plot_destruction_heatmap(output_path=output_path, T=T, S_w=S_W, M=M)
#     pipeline.plot_destruction_AUC(output_path=output_path, time_windows=list(range(0, T - S_W + 1, M)))

def run_pipeline():
    state = {"filepath": "", "w_true": None, "W": None, "graphs": None}

    pipeline = PipelineBuilder(state)
    
    #run_all(pipeline)
    initial_path = "output/testando_2"

    run_scene(pipeline, 1, initial_path, D=11)
    pipeline.execute_pipeline()
    pipeline.pipeline = []

    run_scene(pipeline, 2, initial_path, D=11, drop_w=0.5)
    pipeline.execute_pipeline()
    pipeline.pipeline = []

    run_scene(pipeline, 3, initial_path, D=21, drop_data=0.5)
    pipeline.plot_train_val(partial_filepath=initial_path, scenes=[1, 2, 3], T=100, output_path=initial_path, val=True, train=False, filename="val_errors")
    pipeline.plot_train_val(partial_filepath=initial_path, scenes=[1, 2, 3], T=100, output_path=initial_path, val=False, train=True, filename="train_errors")
    pipeline.plot_train_val(partial_filepath=initial_path, scenes=[1, 2, 3], T=100, output_path=initial_path, val=True, train=True, filename="errors")
    pipeline.execute_pipeline()

if __name__ == "__main__":
    run_pipeline()
