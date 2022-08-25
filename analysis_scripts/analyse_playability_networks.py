import json
from typing import List, Tuple
from itertools import product
from pathlib import Path

import torch as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.mario.plotting import plot_level_from_decoded_tensor

from playability_nets.playability_base import PlayabilityBase
from playability_nets.playability_convnet import PlayabilityConvnet
from playability_nets.playability_mlp import PlayabilityMLP


def get_random_misclassified(
    model: PlayabilityBase, levels: t.Tensor, og_p: t.Tensor, n: int = 8
) -> Tuple[t.Tensor]:
    og_predictions = model(levels).probs.flatten()
    predictions = t.zeros_like(og_predictions)
    predictions[og_predictions > 0.5] = 1.0

    wrongly_classified = levels[predictions != og_p]
    misclassifications = og_predictions[predictions != og_p]
    n_misclassified = len(wrongly_classified)
    range_ = np.random.permutation(n_misclassified)
    random_levels = wrongly_classified[range_[: min(n, n_misclassified)]]
    misclassifications = misclassifications[range_[: min(n, n_misclassified)]]

    return random_levels, misclassifications


def get_val_data() -> List[t.Tensor]:
    df = pd.read_csv("./data/processed/training_levels_results.csv")
    df_levels = df.groupby("level")["marioStatus"].mean()

    print(f"Unique levels: {len(df_levels.index)}")

    levels = []
    playabilities = []
    for l, p in df_levels.iteritems():
        levels.append(json.loads(l))
        playabilities.append(p)

    levels = np.array(levels)[:, :, 1:]
    playabilities = np.array(playabilities)
    # Losing some information.
    playabilities[playabilities > 0.0] = 1.0

    b, h, w = levels.shape
    levels_onehot = np.zeros((b, 11, h, w))
    for batch, level in enumerate(levels):
        for i, j in product(range(h), range(w)):
            c = int(level[i, j])
            levels_onehot[batch, c, i, j] = 1.0

    return [
        t.from_numpy(levels_onehot).type(t.float),
        t.from_numpy(playabilities).type(t.float),
    ]


if __name__ == "__main__":
    # p_net = PlayabilityMLP()
    p_convnet = PlayabilityConvnet()

    # p_net.load_state_dict(
    #     t.load("./models/playability_nets/16354218527979908_mlp_final.pt")
    # )
    p_convnet.load_state_dict(
        t.load(
            "./models/playability_nets/1635948364556223_convnet_w_data_augmentation_w_validation_from_dist_final.pt"
        )
    )

    val_l, val_p = get_val_data()
    wrong_levels, wrong_predictions = get_random_misclassified(p_convnet, val_l, val_p)

    _, axes = plt.subplots(2, 4, figsize=(4 * 7, 2 * 7))
    for level, prediction, ax in zip(wrong_levels, wrong_predictions, axes.flatten()):
        plot_level_from_decoded_tensor(level.reshape(1, 11, 14, 14), ax)

    plt.show()

    # print(wrong_predictions)
