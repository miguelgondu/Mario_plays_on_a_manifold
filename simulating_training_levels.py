"""
This script passes the training levels
through the simulator and gets their
statistics.

In theory, these levels should all be
solvable by the A* agent.
"""
import json
from pathlib import Path
from multiprocessing import Pool

import torch

from train_vae import load_data
from simulator import test_level_from_decoded_tensor
from storage_interface import upload_blob_from_dict


def simulate_level(i, level):
    print(f"Simulating level {i}.")
    simulation_results_path = Path("./data/testing_training_levels")
    simulation_results_path.mkdir(exist_ok=True)

    for j in range(5):
        res = test_level_from_decoded_tensor(level, max_time=30)

        with open(
            simulation_results_path / f"level_{i:04d}_{j}.json",
            "w",
        ) as fp:
            json.dump(res, fp)


def main(processes: int = 5):
    training_tensors, test_tensors = load_data()
    all_levels = torch.cat((training_tensors, test_tensors))

    with Pool(processes=processes) as p:
        p.starmap(simulate_level, enumerate(all_levels))


if __name__ == "__main__":
    main(12)
