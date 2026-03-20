import os
import glob
import pandas as pd
import pickle

subdirectories = [d for d in os.listdir() if os.path.isdir(d) and d.startswith("sub-")]

data_dict = {}
task_items = {}

for subdir in subdirectories:
    tsv_files = glob.glob(os.path.join(subdir, "func", "*.tsv"))
    dfs = {}
    get_task_items = True if len(task_items) == 0 else False
    for tsv_file in tsv_files:
        base_name = os.path.basename(tsv_file)
        name, task, run = base_name.split("_")[0], base_name.split("_")[1], base_name.split("_")[2]
        tr = task + '_' + run
        df = pd.read_csv(tsv_file, sep="\t")
        df = df[df["trial_type"] != "control"]
        df = df.reset_index(drop=True)
        dfs[tr] = df
        if get_task_items:
            task_items[tr] = df[['premise1', 'premise2', 'premise3', 'conclusion', 'trial_type']]
    data_dict[subdir] = dfs

with open("events.pkl", "wb") as f:
    pickle.dump(data_dict, f)

with open("task_items.pkl", "wb") as f:
    pickle.dump(task_items, f)

