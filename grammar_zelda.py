"""
This script implements some grammar checks for Zelda.
"""

import numpy as np


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


def get_doors_in_level(level) -> set:
    x, y = np.where(level == 3)
    if len(x) == 0:
        # There are no doors :(
        return set([])

    # Making sure doors are complete
    left_doors = [(4, 1), (5, 1), (6, 1)]
    right_doors = [(4, 14), (5, 14), (6, 14)]
    upper_doors = [(1, 7), (1, 8)]
    lower_doors = [(9, 7), (9, 8)]
    doors_present_in_level = set([])
    for doors in [left_doors, right_doors, upper_doors, lower_doors]:
        for (xi, yi) in zip(x, y):
            if (xi, yi) in doors:
                is_door_complete = True
                for xj, yj in doors:
                    is_door_complete = is_door_complete and (xj in x) and (yj in y)

                if is_door_complete:
                    doors_present_in_level.add(set(doors))

    return doors_present_in_level


def has_doors_or_stairs(
    level: np.ndarray, possible_door_positions, num_doors: int = 1
) -> bool:
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

    # Making sure doors are complete
    left_doors = [(4, 1), (5, 1), (6, 1)]
    right_doors = [(4, 14), (5, 14), (6, 14)]
    upper_doors = [(1, 7), (1, 8)]
    lower_doors = [(9, 7), (9, 8)]
    doors_present_in_level = set([])
    for doors in [left_doors, right_doors, upper_doors, lower_doors]:
        for (xi, yi) in zip(x, y):
            if (xi, yi) in doors:
                doors_present_in_level.add(tuple(doors))
                for xj, yj in doors:
                    flag_ = flag_ and (xj in x) and (yj in y)

    return flag_ and (len(doors_present_in_level) >= num_doors)


def are_doors_connected(level):
    doors_in_level = get_doors_in_level(level)

    if len(doors_in_level) <= 1:
        return True


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
