from classes.base_class import BaseClass
from lib.logger import success

#---- Experiment Scripts ----
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from scripts.graph_destruction import graph_edge_destruction
from scripts.components_AUC import graph_components_AUC, AUC_interpolation 
# -----------------------

#---- Experiment Plots ----
from lib.functions import plot_graph_destruction_heatmap, plot_error_train_val, plot_weight_CDF
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

    def graph_destruction(self, output_path:str, graphs_state ="graphs", n_components_state = "n_components", W_sorted_state = "W_sorted"):
        step = (
            "Graph Edge Destruction",
            lambda: self.state.update(dict(zip(
                [n_components_state, W_sorted_state], 
                graph_edge_destruction(G=self.state[graphs_state], output_path=output_path))
                )
            ),
        )
        self.pipeline.append(step)

    def plot_destruction_heatmap(self, output_path: str, T: int, S_w: int, M: int, n_components_state = "n_components"):
        step = (
            "Plot Graph Destruction Heatmap",
            lambda: plot_graph_destruction_heatmap(
                n_components=self.state[n_components_state], output_path=output_path, T=T, S_w=S_w, M=M
            )
        )
        self.pipeline.append(step)

    def plot_destruction_AUC(self, output_path: str, time_windows: list, norm_f: NormalizationStrategy, norm_x = False, W_sorted_state = "W_sorted", curves_state = "curves"):
        step = (
                "Plot Graph Components + AUC",
                lambda: self.state.update( 
                    {curves_state:AUC_interpolation(
                    W_sorted=self.state[W_sorted_state], time_windows=time_windows, output_path=output_path, norm_f=norm_f, norm_x=norm_x
                )}
            )
        )
        self.pipeline.append(step)

    def plot_CDF(self, time_windows:list, output_path:str, graphs_state ="graphs"):
        step = (
            "Plot Train and Validation Errors",
            lambda: plot_weight_CDF(output_path=output_path, G=self.state[graphs_state], time_windows=time_windows)
        )
        self.pipeline.append(step)

    def plot_train_val(self, partial_filepath: str, scenes: list, T: int, output_path: str):
        step = (
            "Plot Train and Validation Errors",
            lambda: plot_error_train_val(partial_filepath=partial_filepath, scenes=scenes, T=T, output_path=output_path)
        )
        self.pipeline.append(step)
        

    def normalize_data(self, norm_f: NormalizationStrategy, norm_state: str):
        step = (
            f"Normalize data from {norm_state}",
            lambda: self.state.update(
                {norm_state: norm_f.norm(x = self.state[norm_state] 
                )}
            ),
        )
        self.pipeline.append(step)

    def execute_pipeline(self):
        for index, (step_name, step_action) in enumerate(self.pipeline):
            success(f"Executing step {index + 1}: {step_name}\n")
            step_action()

        success(f"\nPipeline execution completed successfully.")