"""
This script builds the table [ref: table:results:comparing_geometries]
"""
import numpy as np
import pandas as pd

geometries = ["discrete geometry", "continuous geometry", "baselines"]

index = []
for geometry in geometries:
    if geometry != "baselines":
        for m in [100, 200, 300, 400, 500]:
            index.append((geometry, m))

    index.append((geometry, "full"))

index = pd.MultiIndex.from_tuples(index)
columns = [
    (r"$\mathbb{E}[\text{playability}]$", "Interpolation"),
    ("$\mathbb{E}[\text{playability}]$", "Random Walks"),
    ("$\mathbb{E}[\text{diversity}]$", "Interpolation"),
    ("$\mathbb{E}[\text{diversity}]$", "Random Walks"),
]
columns = pd.MultiIndex.from_tuples(columns)
data = np.zeros((len(index), 4))
table = pd.DataFrame(
    data,
    index=index,
    columns=columns,
)

print(table.to_latex(escape=False))
