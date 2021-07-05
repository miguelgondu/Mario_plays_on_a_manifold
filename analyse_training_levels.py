import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from train_vae import load_data
from mario_utils.levels import onehot_to_levels, tensor_to_sim_level
from mario_utils.levels import levels_to_onehot
from simulator import run_level, test_level_from_decoded_tensor

sim_data = Path("./data/testing_training_levels")
all_results = sim_data.glob("*.json")

training_tensors, test_tensors = load_data()

all_levels = torch.cat((training_tensors, test_tensors))
# print(all_levels.shape)

rows = []
for result in all_results:
    with open(result) as fp:
        data = json.load(fp)

    idx = int(result.name.split("_")[1])
    iteration = int(result.name.split("_")[2].replace(".json", ""))
    level = tensor_to_sim_level(all_levels[idx].view(-1, 11, 14, 14))

    row = {"idx": idx, "iteration": iteration, "level": level, **data}

    rows.append(row)

# print(rows)
# print(len(rows))

df = pd.DataFrame(rows)
# print(df)

playable_levels_idx = []
playable_levels = []
for idx in df["idx"].unique():
    df_idx = df[df["idx"] == idx]
    level = json.loads(df_idx["level"].values[0])
    level = np.array(level)[:, 1:]
    # print("level", level, level.shape)

    # print(idx, df_idx["marioStatus"].mean())
    if df_idx["marioStatus"].mean() > 0:
        playable_levels_idx.append(int(idx))
        playable_levels.append(level)

with open("./data/processed/playable_levels_idxs.json", "w") as fp:
    json.dump(playable_levels_idx, fp)

all_playable_levels = np.array(playable_levels, dtype=int)
# print(all_levels[playable_levels_idx])
print(all_levels[playable_levels_idx].shape)
print(all_playable_levels)
print(all_playable_levels.shape)

all_playable_levels = levels_to_onehot(all_playable_levels, n_sprites=11)
np.savez("./data/processed/all_playable_levels_onehot.npz", levels=all_playable_levels)
