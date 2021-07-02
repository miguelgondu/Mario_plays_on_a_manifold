import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from train_vae import load_data
from mario_utils.levels import onehot_to_levels, tensor_to_sim_level
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
print(df)

playable_levels = []
for idx in df["idx"].unique():
    df_idx = df[df["idx"] == idx]
    # if len(df_idx["iteration"].unique()) < 5:
    #     print(df_idx["iteration"].unique())
    #     print(len(df_idx["iteration"].unique()))

    print(idx, df_idx["marioStatus"].mean())
    if df_idx["marioStatus"].mean() > 0:
        playable_levels.append(int(idx))

with open("./data/processed/playable_levels_idxs.json", "w") as fp:
    json.dump(playable_levels, fp)

print(all_levels[playable_levels])
