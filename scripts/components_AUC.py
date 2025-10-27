from lib.functions import plot_AUC
import numpy as np

def graph_components_AUC(W_sorted: np.ndarray, time_windows: list, output_path:str):
    AUC_total = 0
    AUC_partial = 0

    for t in range(0, len(time_windows)):
        Wt = W_sorted[t] # Array of tuples with (weight_value, quantity of components)
        w0 = Wt[0]
        w1 = Wt[len(Wt) - 1]
        
        AUC_partial = (w1[0] - w0[0]) * w0[1]

        plot_AUC(time_windows[t], Wt, output_path)
        AUC_total += AUC_partial