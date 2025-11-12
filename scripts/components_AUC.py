from lib.functions import plot_AUC
import numpy as np
import os
import scipy.interpolate as it
from classes.normalization import NormalizationStrategy

def graph_components_AUC(W_sorted: np.ndarray, time_windows: list, output_path:str, e = 1e-9):
    AUC_total = 0
    AUC_partial = 0
    AUC = np.zeros(len(time_windows))

    for t in range(0, len(time_windows)):
        Wt = W_sorted[t] # Array of tuples with (weight_value, quantity of components)

        before = Wt[0]
        for i in Wt:
            s = (i[0] - before[0])
            if s > e:
                AUC_partial += s * before[1]
                before = i

        plot_AUC(time_windows[t], Wt, AUC_partial, output_path)
        AUC_total += AUC_partial
        AUC[t] = AUC_partial
        AUC_partial = 0

    np.save(os.path.join(output_path, "graph_partial_AUC.npy"), AUC)
    print(AUC_total)

def AUC_interpolation(W_sorted: np.ndarray, time_windows: list, output_path:str, norm_f: NormalizationStrategy, norm_x:bool, delta=0.001):
    areas = []
    
    for t in range(0, len(time_windows)):
        Wt = W_sorted[t]

        x = np.array([i[0] for i in Wt])
        y = np.array([i[1] for i in Wt])
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        if norm_x:
            x = norm_f.norm(x=x)
        y = norm_f.norm(x=y)

        tx = np.arange(min(x), max(x), delta)

        f = it.interp1d(x.flatten(), y.flatten(), kind="nearest")
        ty = map(float, map(f, tx))
        AUC_partial = sum(ty)
        
        areas.append(AUC_partial)
        plot_AUC(time_windows[t], Wt, AUC_partial, output_path) # Vizualize the result

    np.save(os.path.join(output_path, "graph_partial_AUC.npy"), areas)
    print(sum(areas))