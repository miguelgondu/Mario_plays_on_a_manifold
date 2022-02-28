import numpy as np

from grammar_zelda import grammar_check

from utils import get_mean_diversities_of_levels

from train_hierarchical_vae import load_data as load_mario

zelda_levels = np.load("./data/processed/zelda/onehot.npz")["levels"]
zelda_levels = np.array([l for l in zelda_levels if grammar_check(l.argmax(axis=-1))]).argmax(axis=-1)

mario_levels = np.load("./data/processed/all_playable_levels_onehot.npz")[
    "levels"
].argmax(axis=1)

print(f"diversity of zelda: {get_mean_diversities_of_levels(zelda_levels)}")
print(f"diversity of mario: {get_mean_diversities_of_levels(mario_levels)}")
