from classes.base_class import BaseClass
from lib.logger import success
import numpy as np

#---- Experiment Scripts ----
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from scripts.graph_destruction import graph_edge_destruction, edge_destruction
from scripts.components_AUC import graph_components_AUC, AUC_interpolation, AUC_plus_interpolation 
# -----------------------

#---- Experiment Plots ----
from lib.functions import plot_graph_destruction_heatmap, plot_error_train_val, plot_weight_CDF, plot_AUC
# -----------------------

#---- Auxiliar Classes ---
from classes.data_generation import DataGenerationStrategy
from classes.model import ModelStrategy
from classes.algorithm import AlgorithmStrategy
from classes.error_function import ErrorFunctionStrategy
from classes.graph_gen import GraphGenerationStrategy
from classes.weight_association import AssociationStrategy
from classes.normalization import NormalizationStrategy
# -----------------------

class PipelineBuilder(BaseClass):
    
    def __init__(self, state: dict):
        self.pipeline = []
        self.state = state

    def data_generation(self, output_path: str, f_theta: DataGenerationStrategy, r_omega: DataGenerationStrategy, g_lambda: DataGenerationStrategy, N: int, D: int, noise: float, cov=None, drop_w=None, drop_data=None, filepath_state="filepath", w_true_state="w_true"):
        step = (
            "Generating Synthetic Data",
            lambda: (
                self.state.update(dict(zip(
                    [filepath_state, w_true_state],
                    synthetic_data_generation(output_path=output_path, f_theta=f_theta, r_omega=r_omega, g_lambda=g_lambda, N=N, D=D, noise=noise, cov=cov, drop_w=drop_w, drop_data=drop_data)))
                )
            ),
        )
        self.pipeline.append(step)

    def model_training(self, output_path: str, D: int, T: int, lr: float, r_omega: DataGenerationStrategy, e_phi: ErrorFunctionStrategy, H: ModelStrategy, a: AlgorithmStrategy, filepath_state="filepath"):
        step = (
            "Training Process",
            lambda: self.state.update(
                W=training_process(
                    output_path=output_path, filepath=self.state[filepath_state], D=D, T=T, lr=lr, r_omega=r_omega, e_phi=e_phi, H = H, a = a
                )
            ),
        )
        self.pipeline.append(step)

    def graph_generation(self, output_path: str, q: GraphGenerationStrategy, corr: AssociationStrategy, S_w: int, M: int, norm_f: NormalizationStrategy = None, graphs_state = "graphs", W_state = "W"):
        step = (
            "Graph Generation and Graph Plot",
            lambda: self.state.update(
                {graphs_state: generate_graphs(self.state[W_state], output_path=output_path, q=q, corr=corr, S_w=S_w, M=M, norm_f=norm_f
                )}
            ),
        )
        self.pipeline.append(step)

    def graph_destruction(self, output_path:str, filter:np.ndarray, i:int, t:int, x_state:str, y_state:str, graphs_state ="graphs"):
            step = (
                f"Graph Edge Destruction for Time Window {t}",
                lambda: self.state.update(dict(zip(
                    [x_state, y_state], 
                    edge_destruction(G=self.state[graphs_state][i], filter=filter, output_path=output_path, t=t, xt=self.state[x_state], yt=self.state[y_state]))
                    )
                ),
            )
            self.pipeline.append(step)
            

    def plot_destruction_heatmap(self, output_path: str, time_windows: list, x_label: str, AUC_data_output: str):
        step = (
            "Plot Graph Destruction Heatmap",
            lambda: plot_graph_destruction_heatmap(
                output_path=output_path, time_windows=time_windows, x_label=x_label, AUC_data_output=AUC_data_output
            )
        )
        self.pipeline.append(step)

    def calculate_AUC(self, output_path: str, x_state: str, y_state: str, AUC_state:str, t: int, i: int):
        step = (
                f"Interpolation and AUC for Time Window {t}",
                lambda: self.state.update( 
                    {AUC_state: AUC_plus_interpolation(
                    x=self.state[x_state][i], y=self.state[y_state][i], t=t, output_path=output_path, areas=self.state[AUC_state]
                )}
            )
        )
        self.pipeline.append(step)

    def plot_AUC(self, time_window:list, x_label:str, y_label:str, analysis_type:str, output_path: str, AUC_data_output: str, AUC_state: str, t: list):
        step = (
            "Plot AUC",
            lambda: plot_AUC(time_window=time_window, x_label=x_label, y_label=y_label, analysis_type=analysis_type, AUC=self.state[AUC_state], t=t, output_path=output_path, AUC_data_output=AUC_data_output)
        )
        self.pipeline.append(step)

    def plot_CDF(self, time_windows:list, output_path:str, graphs_state ="graphs"):
        step = (
            "Plot CDF of the graph weights",
            lambda: plot_weight_CDF(output_path=output_path, G=self.state[graphs_state], time_windows=time_windows)
        )
        self.pipeline.append(step)

    def plot_train_val(self, partial_filepath: str, scenes: list, T: int, output_path: str, val: bool, train: bool, filename: str):
        step = (
            "Plot Train and Validation Errors",
            lambda: plot_error_train_val(partial_filepath=partial_filepath, scenes=scenes, T=T, output_path=output_path, val=val, train=train, filename=filename)
        )
        self.pipeline.append(step)
        

    def normalize_data(self, norm_f: NormalizationStrategy, norm_state: str, per_line:bool = False):
        step = (
            f"Normalize data from {norm_state}",
            lambda: self.state.update(
                {norm_state: norm_f.norm(x = self.state[norm_state], per_line=per_line 
                )}
            ),
        )
        self.pipeline.append(step)

    def execute_pipeline(self):
        for index, (step_name, step_action) in enumerate(self.pipeline):
            success(f"Executing step {index + 1}: {step_name}\n")
            step_action()

        success(f"\nPipeline execution completed successfully.")