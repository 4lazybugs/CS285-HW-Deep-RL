import os
import wandb

api = wandb.Api()

entity = "lazybugs305-sungkyunkwan-university"
project = "hw1-imitation"

runs = api.runs(f"{entity}/{project}")

os.makedirs("exp", exist_ok=True)

for run in runs:
    run_dir = os.path.join("exp", run.name)
    os.makedirs(run_dir, exist_ok=True)

    for file in run.files():
        file.download(root=run_dir, replace=True)

    print("Downloaded:", run.name)

print("Done.")