from lib.functions import plot_AUC
import numpy as np

def graph_components_AUC(W_sorted: np.ndarray, time_windows: list, output_path:str):
    t_before = 0
    # AUC = 0
    
    for t in range(1, len(time_windows)):
        Wt = W_sorted[t] # Array of tuples with (weight_value, quantity of components)
        
        plot_AUC(time_windows[t], Wt, output_path)