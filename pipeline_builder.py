from classes.base_class import BaseClass
from lib.logger import success

#---- Experiment Scripts ----
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from scripts.graph_destruction import graph_edge_destruction
from scripts.components_AUC import graph_components_AUC
# -----------------------

#---- Experiment Plots ----
from lib.functions import plot_graph_destruction_heatmap
#------------------------

class PipelineBuilder(BaseClass):
    
    def __init__(self, state: dict):
        self.pipeline = []
        self.state = state

    def data_generation(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Generating Synthetic Data",
            lambda: (
                self.state.update(dict(zip(
                    ["filepath", "w_true"],
                    synthetic_data_generation(self.output_path, f_theta=self.f_theta, r_omega=self.r_omega, g_lambda=self.g_lambda, N=self.N, D=self.D, noise=self.noise, cov=self.cov)))
                )
            ),
        )
        self.pipeline.append(step)

    def model_training(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Training Process",
            lambda: self.state.update(
                W=training_process(
                    self.output_path, filepath=self.state["filepath"], D=self.D, T=self.T, lr=self.lr, r_omega=self.r_omega, e_phi=self.e_phi, H = self.H, a = self.a
                )
            ),
        )
        self.pipeline.append(step)

    def graph_generation(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Graph Generation and Graph Plot",
            lambda: self.state.update(
                graphs=generate_graphs(self.state["W"], self.output_path, q=self.q, corr=self.corr, S_w=self.S_w, M=self.M
                )
            ),
        )
        self.pipeline.append(step)

    def graph_destruction(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Graph Edge Destruction",
            lambda: self.state.update(dict(zip(
                ["n_components", "W_sorted"], 
                graph_edge_destruction(G=self.state["graphs"], output_path=self.output_path))
                )
            ),
        )
        self.pipeline.append(step)

    def plot_destruction_heatmap(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Plot Graph Destruction Heatmap",
            lambda: plot_graph_destruction_heatmap(
                n_components=self.state["n_components"], output_path=self.output_path, T=self.T, S_w=self.S_w, M=self.M
            )
        )
        self.pipeline.append(step)

    def plot_destruction_AUC(self, **kwargs):
        self.set_attributes(kwargs)
        step = (
            "Plot Graph Components + AUC",
            lambda: graph_components_AUC(
                W_sorted=self.state["W_sorted"], time_windows=list(range(0, self.T - self.S_w + 1, self.M)), output_path=self.output_path
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