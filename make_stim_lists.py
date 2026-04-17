# make_stim_lists.py
import os
import numpy as np
import pandas as pd
from itertools import permutations

# ==== SETTINGS ====
# Base directory = folder where this script lives
base_dir = os.path.dirname(os.path.abspath(__file__))

# Output folder for premade lists
output_dir = os.path.join(base_dir, "stimuli", "csvs", "maintask_stimlists")
os.makedirs(output_dir, exist_ok=True)

# Factors
categories = ["faces", "places", "fruits"]
operations = ["maintain", "suppress"]   # only 2 operations
probe_types = range(4)
column_order = ["encode_1_cat", "encode_2_cat", "operation", "probe_type"]

# ==== 1. BUILD 48 UNIQUE CONDITIONS ====
base_conditions = []
for op in operations:
    for (cat1, cat2) in permutations(categories, 2):  # ordered pairs
        for probe in probe_types:
            base_conditions.append([cat1, cat2, op, probe])

base_df = pd.DataFrame(base_conditions, columns=column_order)   # (48, 4)

# ==== 2. REPEAT 3× TO GET 144 TRIALS ====
conditions_df = pd.concat([base_df] * 3, ignore_index=True)     # (144, 4)

# ==== 3. BALANCE CUE SIDE WITHIN (CAT1, CAT2, OP) GROUPS ====
conditions_df["cue_position"] = -1
group_cols = ["encode_1_cat", "encode_2_cat", "operation"]

for _, idx in conditions_df.groupby(group_cols).groups.items():
    idx = list(idx)
    n = len(idx)            # should be 12
    cues = np.array([0] * (n // 2) + [1] * (n // 2))  # 6 left, 6 right
    np.random.shuffle(cues)
    conditions_df.loc[idx, "cue_position"] = cues

# ==== 4. ADD RUN NUMBERS: 6 runs × 24 trials ====
run_len = 24
num_runs = 6
conditions_df["run_num"] = np.repeat(np.arange(1, num_runs + 1), run_len)

# ==== 5. SAVE 10 SHUFFLED VERSIONS ====
for i in range(10):
    stim_list = conditions_df.sample(frac=1).reset_index(drop=True)
    fname = os.path.join(output_dir, f"main_stim_list_{i}.csv")
    stim_list.to_csv(fname, index=False)
    print(f"Wrote {fname}")

print("✅ Done creating 10 premade stim lists.")
