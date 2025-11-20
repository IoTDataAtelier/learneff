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

def AUC_interpolation(W_sorted: np.ndarray, time_windows: list, output_path:str, norm_f: NormalizationStrategy, norm_x:bool, delta=0.01):
    areas = []
    #curves = []
    
    for t in range(0, len(time_windows)):
        Wt = W_sorted[t]

        x = np.array([i[0] for i in Wt])
        y = np.array([i[1] for i in Wt])
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        if norm_x:
            x = norm_f.norm(x=x)
        y = norm_f.norm(x=y)

        min_x = min(x)
        max_x = max(x)

        if min_x != max_x:
            tx = np.arange(min_x, max_x, delta)
            tx = np.append(tx, max_x)

            f = it.interp1d(x.flatten(), y.flatten(), kind="nearest")
            ty = f(tx)
            AUC_partial = sum(ty)
            #curves.append(ty)
            plot_AUC(t=[time_windows[t]], x = [tx], y = [ty], output_path=output_path, x_label="a", y_label="b", analysis_type="c") # Vizualize the result
        elif min_x != 1:
            pass
        else:
            AUC_partial = 0
            #curves.append(y.flatten())
            plot_AUC(t=[time_windows[t]], x = [x], y = [y], output_path=output_path, x_label="a", y_label="b", analysis_type="c") # Vizualize the result
        
        areas.append(AUC_partial)

    np.save(os.path.join(output_path, "graph_partial_AUC.npy"), areas)
    print(sum(areas))
    
    #return np.array(curves).T

def AUC_plus_interpolation(x: np.ndarray, y: np.ndarray, t:int, output_path:str, delta=0.01):
    min_x = min(x)
    max_x = max(x)

    if min_x != max_x:
            tx = np.arange(min_x, max_x, delta)
            tx = np.append(tx, max_x)

            f = it.interp1d(x.flatten(), y.flatten(), kind="nearest")
            ty = f(tx)
    else:
        pass

    np.save(os.path.join(output_path, f"graph_AUC_weights_{t}.npy"), tx)
    np.save(os.path.join(output_path, f"graph_AUC_components_{t}.npy"), ty)