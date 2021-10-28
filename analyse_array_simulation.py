import json
from pathlib import Path

import pandas as pd

all_levels = Path("./data/array_simulation_jsons").glob("*_*.json")

rows = []
for i, path in enumerate(all_levels):
    print(f"{i}/{all_levels}\r", flush=True)
    with open(path) as fp:
        row = json.load(fp)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("./data/array_simulation_results/sampling.csv")
print(df)
