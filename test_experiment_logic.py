#!/usr/bin/env python3
"""
Test script for supstress2 experiment logic.
Verifies randomization, stim list generation, and known bug detection.
Run from the supstress directory: python3 test_experiment_logic.py
"""
import os
import sys
import random
import pandas as pd
import numpy as np
from collections import Counter
from itertools import permutations

_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

print("=" * 60)
print("SUPSTRESS2 EXPERIMENT LOGIC TESTS")
print("=" * 60)

# ============================================================
# TEST 1: Stimulus CSV files exist and have correct counts
# ============================================================
print("\n📁 Test Group: Stimulus files")

csv_dir = os.path.join(_thisDir, "stimuli", "csvs", "main_task")
expected_files = [
    "faces_mainTask.csv", "places_mainTask.csv", "fruits_mainTask.csv",
    "faces_novel.csv", "places_novel.csv", "fruits_novel.csv",
]
for f in expected_files:
    path = os.path.join(csv_dir, f)
    exists = os.path.exists(path)
    test(f"Stimulus CSV exists: {f}", exists)
    if exists:
        df = pd.read_csv(path, header=None)
        test(f"  {f} has entries", len(df) > 0, f"got {len(df)} rows")

# Load the pools
faces_main = pd.read_csv(os.path.join(csv_dir, "faces_mainTask.csv"), header=None)[0].tolist()
places_main = pd.read_csv(os.path.join(csv_dir, "places_mainTask.csv"), header=None)[0].tolist()
fruits_main = pd.read_csv(os.path.join(csv_dir, "fruits_mainTask.csv"), header=None)[0].tolist()
faces_novel = pd.read_csv(os.path.join(csv_dir, "faces_novel.csv"), header=None)[0].tolist()
places_novel = pd.read_csv(os.path.join(csv_dir, "places_novel.csv"), header=None)[0].tolist()
fruits_novel = pd.read_csv(os.path.join(csv_dir, "fruits_novel.csv"), header=None)[0].tolist()

print(f"\n  Pool sizes: faces={len(faces_main)}, places={len(places_main)}, fruits={len(fruits_main)}")
print(f"  Novel sizes: faces={len(faces_novel)}, places={len(places_novel)}, fruits={len(fruits_novel)}")

# ============================================================
# TEST 2: Practice stim list
# ============================================================
print("\n📁 Test Group: Practice stim list")

prac_path = os.path.join(_thisDir, "stimuli", "csvs", "maintask_stimlists", "prac_stim_lists.csv")
test("Practice CSV exists", os.path.exists(prac_path))

if os.path.exists(prac_path):
    prac_df = pd.read_csv(prac_path)
    test("Practice CSV has probe_subtype column", "probe_subtype" in prac_df.columns)
    test("Practice CSV has probe_type column", "probe_type" in prac_df.columns)
    test("Practice CSV has 6 trials", len(prac_df) == 6, f"got {len(prac_df)}")
    
    prac_subtypes = prac_df["probe_subtype"].unique().tolist() if "probe_subtype" in prac_df.columns else []
    print(f"  Practice probe_subtypes: {prac_subtypes}")

# ============================================================
# TEST 3: Main task stim list generation (run the actual logic)
# ============================================================
print("\n🔬 Test Group: Main task stim list generation")

random.seed(42)  # Fixed seed for reproducibility

total_trials = 192
run_len = 48
num_runs = 4

categories = ["faces", "places", "fruits"]
operations = ["maintain", "suppress"]
probe_types = range(4)

conditions = [(cat1, cat2, oper, probe)
              for cat1, cat2 in permutations(categories, 2)
              for oper in operations
              for probe in probe_types]
conditions = np.array(conditions)
column_order = ["encode_1_cat", "encode_2_cat", "operation", "probe_type"]

base_df = pd.DataFrame(conditions, columns=column_order)
conditions_df = pd.concat([base_df] * 2, ignore_index=True)
conditions_df["cue_position"] = "left"
conditions_df.loc[48:, "cue_position"] = "right"
conditions_df = pd.concat([conditions_df] * 2, ignore_index=True)

test("Conditions matrix has 192 rows", len(conditions_df) == 192, f"got {len(conditions_df)}")
test("Each condition appears exactly 2x", 
     all(conditions_df.groupby(["encode_1_cat", "encode_2_cat", "operation", "probe_type", "cue_position"]).size() == 2))

# Build runs with operation streak constraint
def max_streak(series):
    grp = (series != series.shift()).cumsum()
    return series.groupby(grp).size().max()

conditions_df = conditions_df.sample(frac=1, random_state=42).reset_index(drop=True)
runs = []
for i in range(num_runs):
    run_df = conditions_df.iloc[i*run_len:(i+1)*run_len].copy()
    for attempt in range(5000):
        run_df = run_df.sample(frac=1).reset_index(drop=True)
        if max_streak(run_df["operation"]) <= 3:
            break
    run_df["run_num"] = i + 1
    runs.append(run_df)

mainTask_df = pd.concat(runs, ignore_index=True)
mainTask_df["trial_num"] = mainTask_df.groupby("run_num").cumcount() + 1
mainTask_df["rest_trigger"] = (mainTask_df["trial_num"] == run_len).astype(int)
mainTask_df.loc[mainTask_df.index[-1], "rest_trigger"] = 0
mainTask_df["jitter"] = 4

test("Generated 192 trials", len(mainTask_df) == 192)
test("4 runs of 48 trials each", 
     all(mainTask_df.groupby("run_num").size() == 48))

# Check operation streak rule
for r in range(1, num_runs + 1):
    rd = mainTask_df[mainTask_df["run_num"] == r]
    streak = max_streak(rd["operation"])
    test(f"Run {r}: max operation streak ≤ 3", streak <= 3, f"got {streak}")

# ============================================================
# TEST 4: Check generated CSV columns vs what Probe code expects
# ============================================================
print("\n🐛 Test Group: BUG DETECTION — probe_subtype mismatch")

generated_columns = list(mainTask_df.columns)
print(f"  Generated stim list columns: {generated_columns}")

has_probe_subtype = "probe_subtype" in generated_columns
test("⚠️  Main stim list has 'probe_subtype' column", has_probe_subtype,
     "MISSING! The Probe routine references probe_subtype but the generated "
     "stim list only has probe_type (numeric 0-3). This will cause the main "
     "task to use stale probe_subtype from the last practice trial, making "
     "correct_ans WRONG for most trials!")

# Map what probe_type values mean
probe_type_map = {
    "0": "cued",
    "1": "uncued", 
    "2": "novel_samecat_cued",
    "3": "novel_diff_cat (uncued cat)"
}
print(f"\n  probe_type mapping in init code:")
for k, v in probe_type_map.items():
    print(f"    {k} → {v}")

print(f"\n  Probe routine expects probe_subtype text values like:")
print(f"    'cued', 'uncued', 'replacement' → correct_ans = 'f' (YES)")
print(f"    anything else → correct_ans = 'j' (NO)")

# ============================================================
# TEST 5: Participant number is hardcoded
# ============================================================
print("\n🐛 Test Group: BUG DETECTION — hardcoded participant_number")

# Read the source and check
with open(os.path.join(_thisDir, "supstress2_maintask_lastrun.py"), "r") as f:
    source = f.read()

# Check for hardcoded participant_number (original bug)
if 'participant_number = "test"' in source:
    # Check if the __main__ fix exists (re-saves with real participant ID after dialog)
    has_main_fix = "participant_number = expInfo['participant']" in source
    if has_main_fix:
        test("participant_number: initial 'test' default is overridden after dialog", True)
    else:
        test("⚠️  participant_number is dynamic (not hardcoded)", False,
             'participant_number = "test" is hardcoded with no override after dialog!')
else:
    test("participant_number is dynamic", True)

# Check that the trial handler path is dynamic
if "sub-test/main_task/main_stim_list.csv" in source:
    test("⚠️  Trial handler path is dynamic (not hardcoded to sub-test)", False,
         "Trial handler loads from hardcoded 'sub-test' path.")
else:
    test("Trial handler path is dynamic", True)

# ============================================================
# TEST 6: Rest screen timer mismatch
# ============================================================
print("\n🐛 Test Group: BUG DETECTION — Rest screen timer")

# The rest routine max duration is 180s but countdown text says 30s
if "180-frameTolerance" in source and '30 - t:.0f' in source:
    test("⚠️  Rest screen: timer text matches actual duration", False,
         "Countdown shows '30 second break' but routine max is 180 seconds. "
         "After 30s the countdown goes negative. SPACE key only works 20-30s.")
else:
    test("Rest screen timer consistency", True)

# ============================================================
# TEST 7: Image randomization across runs
# ============================================================
print("\n🔀 Test Group: Image randomization verification")

# Simulate image assignment like the experiment does
main_pools_master = {
    "faces": faces_main,
    "places": places_main,
    "fruits": fruits_main
}
novel_pools_master = {
    "faces": faces_novel,
    "places": places_novel,
    "fruits": fruits_novel
}

main_pool_sizes = {k: len(v) for k, v in main_pools_master.items()}

def run_encode_demand(run_df):
    return run_df["encode_1_cat"].value_counts().add(
        run_df["encode_2_cat"].value_counts(), fill_value=0
    ).astype(int)

# Check feasibility
all_feasible = True
for r in range(1, num_runs + 1):
    rd = mainTask_df[mainTask_df["run_num"] == r]
    demand = run_encode_demand(rd)
    for cat, needed in demand.items():
        if needed > main_pool_sizes[cat]:
            all_feasible = False
            test(f"Run {r}: {cat} demand ≤ pool size", False, 
                 f"needs {needed}, pool has {main_pool_sizes[cat]}")

test("All runs feasible for no-duplicate encoding", all_feasible)

# Actually assign images and check for duplicates within runs
for run_num in range(1, num_runs + 1):
    run_idx = mainTask_df.index[mainTask_df["run_num"] == run_num].tolist()
    run_df = mainTask_df.loc[run_idx].copy()
    
    main_decks = {cat: random.sample(pools, k=len(pools)) 
                  for cat, pools in main_pools_master.items()}
    
    try:
        e1_imgs = []
        e2_imgs = []
        for _, row in run_df.iterrows():
            e1_imgs.append(main_decks[row["encode_1_cat"]].pop())
            e2_imgs.append(main_decks[row["encode_2_cat"]].pop())
        
        all_encode = e1_imgs + e2_imgs
        dupes = [img for img, count in Counter(all_encode).items() if count > 1]
        test(f"Run {run_num}: no duplicate encode images", len(dupes) == 0,
             f"{len(dupes)} duplicates found")
    except IndexError:
        test(f"Run {run_num}: pool exhaustion (demand > supply)", False,
             "Pool ran empty - this seed produces an infeasible distribution. "
             "The experiment's retry loop handles this, but pool margins are tight.")
        break

# Check images are different ACROSS runs (should reuse from pools)
all_e1 = []
for run_num in range(1, num_runs + 1):
    run_df = mainTask_df[mainTask_df["run_num"] == run_num]
    # Images are drawn from same pool each run, so repeats across runs are expected
    # But within a run, no repeats

# ============================================================
# TEST 8: Correct answer logic verification
# ============================================================
print("\n🎯 Test Group: Correct answer logic")

# What the Probe code does:
# if probe_subtype in ["cued", "uncued", "replacement"]: correct_ans = "f"
# else: correct_ans = "j"
#
# Practice CSV probe_subtypes: cued, uncued, novel_samecatcued, novel_samecatuncued
# Main task probe_types: 0 (cued), 1 (uncued), 2 (novel_samecat), 3 (novel_diff_cat)
#
# For probe_type 0 (cued) → probe IS the cued image → participant saw it → "YES" (f)
# For probe_type 1 (uncued) → probe IS the uncued image → participant saw it → "YES" (f)
# For probe_type 2 (novel_samecat) → novel image → participant did NOT see it → "NO" (j)
# For probe_type 3 (novel_diff_cat) → novel image → participant did NOT see it → "NO" (j)

expected_answers = {"0": "f", "1": "f", "2": "j", "3": "j"}
prac_answers = {"cued": "f", "uncued": "f", "novel_samecatcued": "j", "novel_samecatuncued": "j"}

# Check if the fix is in place (probe_subtype mapping added to init_code)
with open(os.path.join(_thisDir, "supstress2_maintask_lastrun.py"), "r") as f:
    src = f.read()
has_subtype_fix = '_probe_subtype_map' in src and 'mainTask_df["probe_subtype"]' in src
test("FIX: probe_subtype mapping added to stim list generation", has_subtype_fix,
     "Missing _probe_subtype_map code in init_code")

# Since probe_subtype is missing from main task, let's check what would happen
# if the last practice trial's probe_subtype leaks into the main task
if os.path.exists(prac_path):
    prac_df = pd.read_csv(prac_path)
    last_prac_subtype = prac_df.iloc[-1]["probe_subtype"] if "probe_subtype" in prac_df.columns else "unknown"
    
    # The stale value would be used for ALL main task trials
    stale_answer = "f" if last_prac_subtype in ["cued", "uncued", "replacement"] else "j"
    
    print(f"  Last practice trial probe_subtype: '{last_prac_subtype}'")
    print(f"  Stale correct_ans that ALL main trials would use: '{stale_answer}'")
    
    # Count how many main task trials would get wrong correct_ans
    wrong_count = 0
    for _, row in mainTask_df.iterrows():
        pt = str(row["probe_type"])
        actual_correct = expected_answers[pt]
        if stale_answer != actual_correct:
            wrong_count += 1
    
    pct_wrong = wrong_count / len(mainTask_df) * 100
    test(f"⚠️  Main task correct_ans accuracy (with stale probe_subtype)", 
         wrong_count == 0,
         f"{wrong_count}/{len(mainTask_df)} trials ({pct_wrong:.0f}%) would have WRONG "
         f"correct_ans! Every trial would use '{stale_answer}' instead of the real answer.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)

if FAIL > 0:
    print("\n🔴 CRITICAL BUGS FOUND:")
    print("""
1. MISSING probe_subtype IN MAIN TASK STIM LIST
   The Probe routine uses `probe_subtype` to determine correct_ans,
   but the generated main_stim_list.csv only has `probe_type` (numeric).
   Result: ALL main task trials use the STALE probe_subtype from the
   last practice trial, making accuracy scoring completely wrong.
   
   FIX: Add a probe_subtype column to the generated stim list mapping:
     0 → "cued", 1 → "uncued", 2 → "novel_samecatcued", 3 → "novel_samecatuncued"
   
   OR change the Probe code to use probe_type (numeric) instead.

2. HARDCODED participant_number = "test"  
   All participants' stim lists are saved to sub-test/ and the trial
   handler loads from the same hardcoded path.
   
   FIX: Set participant_number = expInfo['participant'] AFTER the
   dialog, or use a "Begin Experiment" code block.

3. REST SCREEN TIMER MISMATCH
   Countdown text says "30 second break" but routine runs for 180s.
   After 30s the countdown goes negative. SPACE key only available 20-30s.
   
   FIX: Either change max duration to 30s, or fix the countdown text
   to match the actual duration.
""")

sys.exit(1 if FAIL > 0 else 0)
