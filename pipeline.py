from datetime import datetime
import os
from lib.logger import success
from scripts.synthetic_data_generation import synthetic_data_generation
from scripts.training_process import training_process
from scripts.graph_generation import generate_graphs
from classes.data_generation import MultivariateGaussian, RandomColumnVector, LinearPlusNoise


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

    state = {"filepath": "", "w_true": None, "W": None}

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
                    output_path, filepath=state["filepath"], D=D, T=T, lr=LR, r_omega=RandomColumnVector()
                )
            ),
        ),
        (
            "Graph Generation",
            lambda: generate_graphs(state["W"], output_path, S_w=S_W, M=M),
        ),
    ]

    for index, (step_name, step_action) in enumerate(pipeline_steps):
        success(f"Executing step {index + 1}: {step_name}\n")
        step_action()

    success(f"\nPipeline execution completed successfully. Results stored in: {output_path}")


if __name__ == "__main__":
    run_pipeline()
