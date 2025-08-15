import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

model_name = sys.argv[1]
val_set = sys.argv[2]

results_path = PROJECT_ROOT / "data/results/grid_search" / model_name / val_set
results_file = results_path / f"{val_set}_consolidated_results.json"
config_file = PROJECT_ROOT / "configs/grid_combinations.json"
output_file = results_path / "missing_indices.txt"

with open(results_file, 'r') as f:
    results = json.load(f)
df = pd.DataFrame(results)

with open(config_file, 'r') as f:
    config = json.load(f)

if len(df) == 3600:
    print("All combinations present, nothing missing.")
    exit(0)

missing_indices = [
    i for i, cfg in enumerate(config)
    if df[(df['ckpt'] == cfg['checkpoint_number']) &
          (df['head'] == cfg['head']) &
          (df['layer'] == cfg['layer'])].empty
]

print(f"{len(missing_indices)} missing results found, saving to {output_file}")
with open(output_file, 'w') as f:
    for idx in missing_indices:
        f.write(f"{idx}\n")

