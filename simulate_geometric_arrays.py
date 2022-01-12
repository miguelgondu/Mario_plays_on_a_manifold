"""
This script simulates all the arrays that need
simulating for the baseline: all the linear interpolations
and all the diffusions (normal and baseline).
"""
from typing import List

from pathlib import Path

# Models and their dimensions
from baseline_metrics_for_different_z_dims import models
from simulate_array import _simulate_array


def get_all_array_paths(model_name, skip_simulated=False) -> List[str]:
    # Clean previously simulated stuff
    if skip_simulated:
        res_path = Path("./data/array_simulation_results/geometric_gpc")
        already_simulated = [
            s.name.replace(".csv", ".npz") for s in res_path.glob("*.csv")
        ]
    else:
        already_simulated = []

    paths = [
        str(s)
        for s in Path("./data/arrays/geometric_gpc").glob(f"{model_name}_*.npz")
        if s.name not in already_simulated
    ]

    return paths


if __name__ == "__main__":
    # for _, model_name in models.items():
    #     all_paths_to_simulate = get_all_array_paths(model_name)
    #     print(f"Simulating {len(all_paths_to_simulate)} files")
    #     for path in all_paths_to_simulate:
    #         print(f"Simulating array {path}")
    #         _simulate_array(path, 32, 5, exp_folder="geometric")

    # Simulating only the 2D for now.
    all_paths_to_simulate = get_all_array_paths(models[2])
    print(f"Simulating {len(all_paths_to_simulate)} files")
    for path in all_paths_to_simulate:
        print(f"Simulating array {path}")
        _simulate_array(path, 32, 5, exp_folder="geometric_gpc")
