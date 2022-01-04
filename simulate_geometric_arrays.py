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


def get_all_array_paths(model_name) -> List[str]:
    # Clean previously simulated stuff
    res_path = Path("./data/array_simulation_results/geometric")
    already_simulated = [s.name.replace(".csv", ".npz") for s in res_path.glob("*.csv")]

    paths = [
        str(s)
        for s in Path("./data/arrays/geometric").glob(f"{model_name}_*.npz")
        if s.name not in already_simulated
    ]

    return paths


if __name__ == "__main__":
    for _, model_name in models.items():
        for path in get_all_array_paths(model_name):
            print(f"Simulating array {path}")
            _simulate_array(path, 32, 5, exp_folder="geometric")
