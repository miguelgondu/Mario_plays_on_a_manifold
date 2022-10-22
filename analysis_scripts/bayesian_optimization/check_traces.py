"""
Given traces of Vanilla, Constrained and Graph-based Bayesian Optimization,
this script loads them and checks metrics regarding nr. of playable levels
and best playable level.
"""

import numpy as np

if __name__ == "__main__":
    arr_vanilla_bo = np.load("./data/bayesian_optimization/traces/vanilla_bo.npz")
    arr_constrained_bo = np.load(
        "./data/bayesian_optimization/traces/constrained_bo.npz"
    )

    p_vanilla = arr_vanilla_bo["playability"]
    p_constrained = arr_constrained_bo["playability"]

    print(f"vanilla: {sum(p_vanilla)}/{len(p_vanilla)}")
    print(f"constrained: {sum(p_constrained)}/{len(p_constrained)}")

    valid_jumps_vanilla = arr_vanilla_bo["jumps"][p_vanilla == 1]
    valid_jumps_constrained = arr_constrained_bo["jumps"][p_constrained == 1]
    print(f"jumps vanilla: {max(valid_jumps_vanilla)}")
    print(f"jumps constrained: {max(valid_jumps_constrained)}")
