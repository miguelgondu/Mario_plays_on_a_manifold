"""
This script passes the training levels
through the simulator and gets their
statistics.

In theory, these levels should all be
solvable by the A* agent.
"""
import json
from multiprocessing import Pool

import torch

from train_vae import load_data
from simulator import test_level_from_decoded_tensor
from storage_interface import upload_blob_from_dict


def simulate_level(i, level):
    print(f"Simulating level {i}.")
    for j in range(5):
        res = test_level_from_decoded_tensor(level, max_time=30)

        with open(
            f"./data/testing_training_levels/level_{i:04d}_{j}.json",
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
