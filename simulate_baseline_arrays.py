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
    li_paths = [
        f"./data/arrays/baselines/{model_name}_linear_interpolation_{line_i:03d}.npz"
        for line_i in range(20)
    ]
    nd_paths = [
        f"./data/arrays/baselines/{model_name}_normal_diffusion_{run_i:03d}.npz"
        for run_i in range(20)
    ]
    bd_paths = [
        f"./data/arrays/baselines/{model_name}_baseline_diffusion_{run_i:03d}.npz"
        for run_i in range(20)
    ]

    return li_paths + nd_paths + bd_paths


if __name__ == "__main__":
    for _, model_name in models.items():
        for path in get_all_array_paths(model_name):
            print(f"Simulating array {path}")
            _simulate_array(path, 5, 5, exp_folder="baselines")
