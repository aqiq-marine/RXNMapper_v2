import json

grid = []
checkpoints = list(range(0,1500001, 10000))
layers = [9, 10, 11]
heads = range(8)

for ckpt in checkpoints:
    for layer in layers:
        for head in heads:
            grid.append({
                "checkpoint_number": str(ckpt),
                "layer": layer,
                "head": head
            })

with open("configs/grid_combinations.json", "w") as f:
    json.dump(grid, f, indent=2)