"""
This script implements some grammar checks for Zelda.
"""

import numpy as np

from zelda_utils.plotting import encoding


def grammar_check(level: np.ndarray) -> bool:
    onehot = np.load("./data/processed/zelda/onehot.npz")["levels"]
    levels = onehot.argmax(axis=-1)
    _, x, y = np.where(levels == 3)
    possible_door_positions = set([(xi, yi) for xi, yi in zip(x, y)])

    flag_ = has_outer_walls(level)
    flag_ = flag_ and has_doors_or_stairs(level, possible_door_positions)

    return flag_


def has_outer_walls(level: np.ndarray) -> bool:
    flag_ = (level[:, 0] == 13).all()
    flag_ = flag_ and (level[:, -1] == 13).all()
    flag_ = flag_ and (level[0, :] == 13).all()
    flag_ = flag_ and (level[-1, :] == 13).all()
    return flag_


def has_doors_or_stairs(level: np.ndarray, possible_door_positions) -> bool:
    """
    Checks if it has doors in the right places
    """
    stairs_x, _ = np.where(level == 10)
    if len(stairs_x) > 0:
        # there are stairs
        return True

    x, y = np.where(level == 3)
    if len(x) == 0:
        # There are no doors :(
        return False

    # Check if the doors are in sensible positions
    flag_ = True
    for xi, yi in zip(x, y):
        flag_ = flag_ and ((xi, yi) in possible_door_positions)

    return flag_


if __name__ == "__main__":
    onehot = np.load("./data/processed/zelda/onehot.npz")["levels"]
    levels = onehot.argmax(axis=-1)

    print(levels)
    b, x, y = np.where(levels == 3)
    print(np.where(levels == 3))
    for level in levels:
        if not grammar_check(level):
            print(level)

    print(len([l for l in levels if not grammar_check(l)]))
