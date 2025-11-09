from lib.functions import plot_AUC
import numpy as np
import os

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

def AUC_interpolation(W_sorted: np.ndarray, time_windows: list, output_path:str, delta=0.001):
    areas = []
    tx = np.arange(0, 1 + delta, delta)
    
    for t in range(0, len(time_windows)):
        Wt = W_sorted[t]

        x = [i[0] for i in Wt]
        y = [i[1] for i in Wt]
        x.append(1.0)
        y.append(y[len(y) - 1])

        idx = np.searchsorted(x, tx, side="left")
        idx = np.clip(idx, 0, len(x) - 1)
        AUC_partial = sum(idx)
        
        areas.append(AUC_partial)
        plot_AUC(time_windows[t], Wt, AUC_partial, output_path) # Vizualize the result

    np.save(os.path.join(output_path, "graph_partial_AUC.npy"), areas)
    print(sum(areas))