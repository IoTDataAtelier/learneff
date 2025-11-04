from classes.base_class import BaseClass

#---- Experiment Phases ----
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from scripts.graph_destruction import graph_edge_destruction
from scripts.components_AUC import graph_components_AUC
# -----------------------

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