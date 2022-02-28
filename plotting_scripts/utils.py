from pathlib import Path

import numpy as np

from experiment_utils import load_csv_as_arrays

def loading_results():
    """
    Gets the csvs and returns points, results and levels
    """
    res_path = Path("./data/array_simulation_results/ten_vaes/final_plots")
    array_path = Path("./data/arrays/ten_vaes/final_plots")
    zs_b, p_b = load_csv_as_arrays(res_path / f"banner_plot_baseline.csv")
    zs_c, p_c = load_csv_as_arrays(res_path / f"banner_plot_continuous.csv")
    zs_d, p_d = load_csv_as_arrays(res_path / f"banner_plot_discrete.csv")

    zs_b_prime = np.load(array_path / f"banner_plot_baseline.npz")["zs"]
    levels_b = np.load(array_path / f"banner_plot_baseline.npz")["levels"]
    zs_c_prime = np.load(array_path / f"banner_plot_continuous.npz")["zs"]
    levels_c = np.load(array_path / f"banner_plot_continuous.npz")["levels"]
    zs_d_prime = np.load(array_path / f"banner_plot_discrete.npz")["zs"]
    levels_d = np.load(array_path / f"banner_plot_discrete.npz")["levels"]

    # Instead, I should just reorder the ps according to zs_c_prime
    p_b = [p_b[zs_b.tolist().index(z.tolist())] for z in zs_b_prime]
    p_c = [p_c[zs_c.tolist().index(z.tolist())] for z in zs_c_prime]
    p_d = [p_d[zs_d.tolist().index(z.tolist())] for z in zs_d_prime]

    return (
        zs_b_prime,
        levels_b,
        p_b,
        zs_c_prime,
        levels_c,
        p_c,
        zs_d_prime,
        levels_d,
        p_d,
    )