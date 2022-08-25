"""
This data grabs the levels from ./data/raw
and splits them into 14x14 chunks. The
original 16x16 were used because of convolutions,
which is not necessary in our case.
"""
import json
from pathlib import Path
from typing import List
from itertools import product

import numpy as np

from utils.mario.levels import levels_to_onehot
from utils.mario.plotting import save_level_from_array


def create_level_array_from_rows(strings: List[str]):
    """
    Returns array with level.
    """
    rows = [list(string.replace("\n", "")) for string in strings]
    return np.array(rows)


def array_to_string_rows(level: np.ndarray):
    level_ = level.tolist()
    level_ = ["".join(r) + "\n" for r in level_]
    return level_


def process(width=14, comment=""):
    """
    Slices all levels into 14x14 chunks.

    Saves .npzs with all the levels.
    """
    data_path = Path("./data")
    raw_level_paths = (data_path / "raw").glob("*.txt")
    processed_level_path = data_path / "processed" / "text"
    processed_level_path.mkdir(exist_ok=True)
    level_slices = []
    for path in raw_level_paths:
        with open(path) as fp:
            rows_level = fp.readlines()
            entire_level = create_level_array_from_rows(rows_level)
            entire_level[entire_level == "b"] = "-"
            entire_level[entire_level == "B"] = "-"

        _, level_length = entire_level.shape
        for i in range(level_length - width):
            level_slice = entire_level[:, i : i + width]
            level_slices.append(level_slice)

    for i, level_slice in enumerate(level_slices):
        strings = array_to_string_rows(level_slice)
        strings[-1].replace("\n", "")
        string = "".join(strings)
        with open(processed_level_path / f"{comment}{i:05d}.txt", "w") as fp:
            fp.write(string)

    level_slices = np.array(level_slices)
    np.savez(
        data_path / "processed" / f"{comment}all_levels_text.npz", levels=level_slices
    )

    with open("./encoding.json") as fp:
        encoding = json.load(fp)

    for text, enc in encoding.items():
        level_slices[level_slices == text] = str(enc)

    # Saving as classes
    level_slices = level_slices.astype(int)
    np.savez(
        data_path / "processed" / f"{comment}all_levels_encoded.npz",
        levels=level_slices,
    )

    # Saving as onehot
    level_slices = levels_to_onehot(level_slices, n_sprites=len(encoding.keys()))
    np.savez(
        data_path / "processed" / f"{comment}all_levels_onehot.npz", levels=level_slices
    )


def process_zelda():
    """
    Loads and processes the Zelda data.
    """
    data_path = Path("./data")
    raw_level_paths = (data_path / "raw" / "og_zelda_levels").rglob("*.txt")
    processed_level_path = data_path / "processed" / "zelda"
    processed_level_path.mkdir(exist_ok=True)

    all_levels = []
    for p in raw_level_paths:
        with open(p) as fp:
            rows_level = fp.readlines()
            entire_level = create_level_array_from_rows(rows_level)
            all_levels.append(entire_level)

    levels = np.array(all_levels)
    np.savez(processed_level_path / "all_levels_text.npz", levels=levels)

    tokens = np.unique(levels)
    encoding = {t: i for i, t in enumerate(tokens)}

    n_levels = len(levels)
    h, w = levels.shape[1], levels.shape[2]
    n_sprites = len(encoding)
    onehot_levels = np.zeros((n_levels, h, w, n_sprites))
    for n, lvl in enumerate(levels):
        for i, j in product(range(h), range(w)):
            _id = encoding[lvl[i, j]]
            onehot_levels[n, i, j, _id] = 1.0

    # Saving levels as .npz and encoding as .json
    with open(processed_level_path / "encoding.json", "w") as fp:
        json.dump(encoding, fp)

    np.savez(processed_level_path / "onehot.npz", levels=onehot_levels)


def plot_all_levels():
    """
    Loads all the training levels and plots them
    """
    levels_path = Path("./data/processed/all_levels_onehot.npz")
    levels = np.load(levels_path)["levels"].argmax(axis=1)

    plotting_path = Path("./data/plots/all_training_levels")
    plotting_path.mkdir(exist_ok=True, parents=True)

    for i, lvl in enumerate(levels):
        save_level_from_array(plotting_path / f"{i:04d}.png", lvl)


if __name__ == "__main__":
    # process()
    # process(width=16, comment="rasmus_")
    # all_levels_encoded = np.load("./data/processed/all_levels_encoded.npz")["levels"]
    # print("Amount of levels: ")
    # print(all_levels_encoded.shape)
    # process_zelda()
    plot_all_levels()
