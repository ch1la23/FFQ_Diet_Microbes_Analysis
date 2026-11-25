"""
Step 0: check that the input files and the settings in config.py fit together.

Run this first (run_pipeline.py does it automatically). It stops with a
readable message instead of letting a typo in config.py turn into a
confusing error halfway through the analysis.

    python scripts/00_check_inputs.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # config.py
sys.path.insert(0, str(HERE))           # ibdlib.py, engines.py

import pandas as pd

import config as C
import ibdlib as L


problems = []
notes = []


def problem(text):
    problems.append(text)
    print("  [PROBLEM]", text)


def ok(text):
    print("  [ok]     ", text)


print("Checking config.py ...")
config_problems = C.check_config()
for p in config_problems:
    problem(p)
if not config_problems:
    ok("settings are consistent")

# ---- metadata -------------------------------------------------------
print("\nChecking the metadata file ...")
meta = None
try:
    meta = L.load_metadata()
    ok(f"{C.METADATA_FILE}: {len(meta)} samples after keeping "
       f"{C.DATA_TYPE_COLUMN} == {C.DATA_TYPE_VALUE!r}" if C.DATA_TYPE_COLUMN else
       f"{C.METADATA_FILE}: {len(meta)} samples")
except Exception as err:
    problem(str(err))

if meta is not None:
    if len(meta) == 0:
        problem("no samples left after the data-type filter; check DATA_TYPE_COLUMN / DATA_TYPE_VALUE")

    for internal, column in C.SAMPLE_COLUMNS.items():
        if column not in meta.columns:
            optional = internal in ("sex", "week_num", "fecalcal")
            msg = f"metadata column '{column}' (for '{internal}') not found"
            if optional:
                print(f"  [note]    {msg} -> optional, will be skipped")
            else:
                problem(msg)

    # diagnosis labels
    diag_col = C.SAMPLE_COLUMNS["diagnosis"]
    if diag_col in meta.columns:
        present = set(meta[diag_col].dropna().unique())
        missing = [g for g in C.DIAG_LEVELS if g not in present]
        if missing:
            problem(f"DIAG_LEVELS has groups that are not in the data: {missing} "
                    f"(found: {sorted(present)})")
        else:
            ok(f"diagnosis groups found: {C.DIAG_LEVELS}")

    # FFQ items
    print("\nChecking the dietary items ...")
    try:
        cols = L.food_columns(meta)
        ok(f"all {len(cols)} FOODS matched to a metadata column")
        for short, col in cols.items():
            answers = meta[col].dropna().unique()
            if C.FFQ_SCORES is not None:
                unknown = [a for a in answers if a not in C.FFQ_SCORES]
                if unknown:
                    problem(f"{short}: answers not in FFQ_SCORES: {unknown}")
            else:
                numeric = pd.to_numeric(pd.Series(answers), errors="coerce")
                if numeric.isna().any():
                    problem(f"{short}: FFQ_SCORES is None but '{col}' has non-numeric answers")
        if not any("answers not in" in p or "non-numeric" in p for p in problems):
            ok("every FFQ answer can be scored")
    except Exception as err:
        problem(str(err))

# ---- abundance ------------------------------------------------------
print("\nChecking the abundance table ...")
try:
    ab, source = L.load_abundance(log=lambda *a, **k: None)
    ok(f"{source}: {ab.shape[0]} samples x {ab.shape[1]} species")
    if meta is not None:
        shared = ab.index.intersection(meta.index)
        if len(shared) == 0:
            problem("no sample IDs are shared between the abundance table and the metadata "
                    f"(metadata IDs come from '{C.SAMPLE_ID_COLUMN}')")
        else:
            ok(f"{len(shared)} samples are in both files")
    if (ab < 0).any().any():
        problem("abundance table has negative values")
except Exception as err:
    problem(str(err))

# ---- result ---------------------------------------------------------
print()
if problems:
    print(f"{len(problems)} problem(s) found. Fix them in config.py or data/ and run again.")
    sys.exit(1)
print("All checks passed.")
