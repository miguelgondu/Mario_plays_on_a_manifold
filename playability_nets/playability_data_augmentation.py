"""
This script constructs several levels that are
non-playable by sliding columns and gaps in many
permutations

From experimentation:
- columns that are taller than 4 are non-playable.
- gaps that are wider than 5 are non-playable.
"""
import random
from typing import List, Tuple
import numpy as np
from utils.mario.levels import clean_level

from simulator import run_level

# Taken from the cookbook in itertools' documentation
def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


# Taken from the cookbook in itertools' documentation
def random_product(*args, repeat=1):
    "Random selection from itertools.product(*args, **kwds)"
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(map(random.choice, pools))


def get_random_column_specifications() -> Tuple[List[int]]:
    """
    Selects a random combination (14 k) and random heights for a random k
    between 1 and 4.

    Returns column positions and heights.
    """
    k = random.randint(1, 4)
    column_positions = random_combination(range(0, 14), k)
    column_heights = random_product(range(5, 7), repeat=len(column_positions))

    return column_positions, column_heights


def get_random_pipes_specifications() -> Tuple[List[int]]:
    """
    Selects a random combination (14 k) and random heights for a random k
    between 1 and 4.

    Returns pipe positions and heights.
    """
    k = random.randint(1, 3)
    pipe_positions = random_combination(range(0, 13), k)
    pipe_heights = random_product(range(5, 7), repeat=len(pipe_positions))

    return pipe_positions, pipe_heights


def get_random_gap_specifications() -> Tuple[int]:
    """
    Returns a random gap interval of length at least 6.
    """
    gap_length = random.randint(6, 14)
    lower = random.randint(0, 14 - gap_length)
    upper = lower + gap_length

    return (lower, upper)


def get_more_non_playable_levels(n_levels, seed=None) -> np.ndarray:
    # Right now I'm only creating columns with stone. It might
    # be better to add columns with more non-passable blocks. The
    # network might overfit to thinking that stones == bad.
    if seed is not None:
        random.seed(seed)

    basic_level = 2.0 * np.ones((14, 14))
    basic_level[-1, :] = 0.0
    levels = []
    for _ in range(n_levels):
        level = basic_level.copy()

        # Adding columns
        column_pos, column_h = get_random_column_specifications()
        for pos, h in zip(column_pos, column_h):
            level[-h - 1 :, pos] = 0.0

        # Adding pipes
        pipe_pos, pipe_h = get_random_pipes_specifications()
        for pos, h in zip(pipe_pos, pipe_h):
            level[-h:-1, pos] = 8.0
            level[-h:-1, pos + 1] = 9.0
            level[-h - 1, pos] = 6.0
            level[-h - 1, pos + 1] = 7.0

        if random.choice((True, False)):
            lower, upper = get_random_gap_specifications()
            level[-1, lower:upper] = 2.0

        levels.append(level)

    levels = np.stack(levels).astype(int)

    return levels


if __name__ == "__main__":
    levels = get_more_non_playable_levels(2000)

    i = random.randint(0, len(levels) - 1)
    level = levels[i]

    run_level(str(clean_level(level)), human_player=True, max_time=10)
