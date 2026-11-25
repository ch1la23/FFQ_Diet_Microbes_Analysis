"""
Step 1: build the analysis tables and an audit of how many samples
each filtering step removes.

Writes (results/D4_Reproducible_Pipeline/):
    analysis_tables/analysis_samples.csv     one row per sample, covariates + food scores
    analysis_tables/analysis_abundance.csv   species abundance for the same samples
    tables/audit.csv                         sample / participant counts per step
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # config.py
sys.path.insert(0, str(HERE))           # ibdlib.py, engines.py

import pandas as pd

import config as C
import ibdlib as L

C.make_output_dirs()


def add_audit_row(audit, label, samples, subjects):
    audit.append((label, len(samples), subjects.loc[samples].nunique()))


metadata = L.load_metadata()
abundance, abundance_source = L.load_abundance()
foods = L.encode_foods(metadata)

subject_col = C.SAMPLE_COLUMNS["Subject"]
audit = []
add_audit_row(audit, "samples in metadata (after data-type filter)",
              metadata.index, metadata[subject_col])

# samples that are in both the metadata and the abundance table
abundance = abundance.loc[abundance.index.intersection(metadata.index)]
add_audit_row(audit, "samples with a species-level profile",
              abundance.index, metadata.loc[abundance.index, subject_col])

# drop samples with no abundance at all
abundance = abundance.loc[abundance.sum(axis=1) > 0]
add_audit_row(audit, "samples with non-zero abundance over the species panel",
              abundance.index, metadata.loc[abundance.index, subject_col])

# sample table, using the pipeline's short column names
wanted = {internal: col for internal, col in C.SAMPLE_COLUMNS.items() if col in metadata.columns}
samples = (
    metadata.loc[abundance.index, list(wanted.values())]
    .rename(columns={col: internal for internal, col in wanted.items()})
    .join(foods)
)

samples["Site"] = samples["Site"].astype(str).str.replace(" ", "_")
samples = samples[samples["diagnosis"].isin(C.DIAG_LEVELS)]

# complete covariates
covariates = list(dict.fromkeys(C.NUMERIC_COVARIATES + C.CATEGORICAL_COVARIATES))
has_covariates = samples[covariates].notna().all(axis=1)
add_audit_row(audit, "with complete covariates (" + ", ".join(covariates) + ")",
              samples.index[has_covariates], samples["Subject"])

# complete food scores
has_all_foods = samples[list(C.FOODS)].notna().all(axis=1)
keep_sample = has_covariates & has_all_foods
add_audit_row(audit, f"with all {len(C.FOODS)} food items",
              samples.index[keep_sample], samples["Subject"])

samples = samples.loc[keep_sample].copy()

if samples.empty:
    sys.exit("No samples left after filtering. Run scripts/00_check_inputs.py to see why.")

# each person's mean intake and the visit's deviation from it
samples = L.add_person_mean_dev(samples, list(C.FOODS))

# keep matching abundance rows, renormalise to sum to 1
abundance = abundance.loc[samples.index]
abundance = abundance.div(abundance.sum(axis=1), axis=0)

add_audit_row(audit, "FINAL analysis set", samples.index, samples["Subject"])

audit_table = pd.DataFrame(audit, columns=["step", "samples", "participants"])
audit_table.to_csv(C.D4_TAB_DIR / "audit.csv", index=False)
samples.to_csv(C.ANALYSIS_DIR / "analysis_samples.csv")
abundance.to_csv(C.ANALYSIS_DIR / "analysis_abundance.csv")

print(f"Abundance source: {abundance_source}")
print(audit_table.to_string(index=False))
print("Participants by diagnosis:",
      samples.groupby("diagnosis")["Subject"].nunique().to_dict())
print(f"Species: {abundance.shape[1]} | Zeros: {abundance.eq(0).to_numpy().mean() * 100:.1f}%")
