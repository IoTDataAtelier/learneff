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
from lib.functions import plot_graph_destruction_heatmap
# -----------------------

#---- Auxiliar Classes ---
from classes.data_generation import DataGenerationStrategy
from classes.model import ModelStrategy
from classes.algorithm import AlgorithmStrategy
from classes.error_function import ErrorFunctionStrategy
from classes.graph_gen import GraphGenerationStrategy
from classes.weight_association import AssociationStrategy
# -----------------------

class PipelineBuilder(BaseClass):
    
    def __init__(self, state: dict):
        self.pipeline = []
        self.state = state

    def data_generation(self, output_path: str, f_theta: DataGenerationStrategy, r_omega: DataGenerationStrategy, g_lambda: DataGenerationStrategy, N: int, D: int, noise: float, cov=None, drop_w=None, drop_data=None):
        step = (
            "Generating Synthetic Data",
            lambda: (
                self.state.update(dict(zip(
                    ["filepath", "w_true"],
                    synthetic_data_generation(output_path=output_path, f_theta=f_theta, r_omega=r_omega, g_lambda=g_lambda, N=N, D=D, noise=noise, cov=cov, drop_w=drop_w, drop_data=drop_data)))
                )
            ),
        )
        self.pipeline.append(step)

    def model_training(self, output_path: str, D: int, T: int, lr: float, r_omega: DataGenerationStrategy, e_phi: ErrorFunctionStrategy, H: ModelStrategy, a: AlgorithmStrategy):
        step = (
            "Training Process",
            lambda: self.state.update(
                W=training_process(
                    output_path=output_path, filepath=self.state["filepath"], D=D, T=T, lr=lr, r_omega=r_omega, e_phi=e_phi, H = H, a = a
                )
            ),
        )
        self.pipeline.append(step)

    def graph_generation(self, output_path: str, q: GraphGenerationStrategy, corr: AssociationStrategy, S_w: int, M: int, graphs_state = "graphs"):
        step = (
            "Graph Generation and Graph Plot",
            lambda: self.state.update(
                {f"{graphs_state}": generate_graphs(self.state["W"], output_path=output_path, q=q, corr=corr, S_w=S_w, M=M
                )}
            ),
        )
        self.pipeline.append(step)

    def graph_destruction(self, output_path:str, graphs_state ="graphs", n_components_state = "n_components", W_sorted_state = "W_sorted"):
        step = (
            "Graph Edge Destruction",
            lambda: self.state.update(dict(zip(
                [f"{n_components_state}", f"{W_sorted_state}"], 
                graph_edge_destruction(G=self.state[f"{graphs_state}"], output_path=output_path))
                )
            ),
        )
        self.pipeline.append(step)

    def plot_destruction_heatmap(self, output_path: str, T: int, S_w: int, M: int, n_components_state = "n_components"):
        step = (
            "Plot Graph Destruction Heatmap",
            lambda: plot_graph_destruction_heatmap(
                n_components=self.state[f"{n_components_state}"], output_path=output_path, T=T, S_w=S_w, M=M
            )
        )
        self.pipeline.append(step)

    def plot_destruction_AUC(self, output_path: str, time_windows: list, W_sorted_state = "W_sorted"):
        step = (
            "Plot Graph Components + AUC",
            lambda: AUC_interpolation(
                W_sorted=self.state[f"{W_sorted_state}"], time_windows=time_windows, output_path=output_path
            )
        )
        self.pipeline.append(step)

    def plot_CDF(self, **kwargs):
        self.set_attributes(kwargs)
        #self.pipeline.append(step)
        pass

    def execute_pipeline(self):
        for index, (step_name, step_action) in enumerate(self.pipeline):
            success(f"Executing step {index + 1}: {step_name}\n")
            step_action()

        success(f"\nPipeline execution completed successfully.")