import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

from vae_geometry import VAEGeometry

# Types
Tensor = torch.Tensor

model_name = "mariovae_z_dim_2_overfitting_epoch_480_playability_experiment"

# Getting the playable levels
# Table was created in analyse_solvability_experiment.py
def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}.csv", index_col=0
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points
