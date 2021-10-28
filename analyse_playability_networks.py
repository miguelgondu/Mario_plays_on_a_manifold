import json
from typing import List
from itertools import product
from pathlib import Path

import torch as t
import pandas as pd
import numpy as np

from playability_convnet import PlayabilityConvnet
from playability_net import PlayabilityNet, get_level_datasets


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
    p_net = PlayabilityNet()
    p_convnet = PlayabilityConvnet()

    p_net.load_state_dict(t.load("./models/playability_net/model_final.pt"))
    p_convnet.load_state_dict(t.load("./models/playability_convnet/model_final.pt"))

    val_l, val_p = get_val_data()
    pred_net = p_net(val_l)
    pred_convnet = p_convnet(val_l)

    pass
