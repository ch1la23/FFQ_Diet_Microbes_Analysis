"""
config.py -- the one place to change settings for the whole pipeline.

Every script reads from here, so if you want to run the pipeline on
your own data, or change the diet items / scoring, you should only need
to edit this file (and drop your input files in data/).

Sections
    1. Input files and where outputs go
    2. Metadata column names
    3. FFQ (food frequency questionnaire) scoring
    4. Dietary items
    5. Covariates and groups
    6. Statistical settings
    7. Model specifications
    8. Helper functions (BH correction, config checks)
"""

import os
from pathlib import Path

import numpy as np


# ======================================================================
# 1. INPUT FILES AND OUTPUT FOLDERS
# ======================================================================

ROOT = Path(__file__).resolve().parent

# All input files live in this folder (can be overridden with an
# environment variable, e.g.  IBD_DATA_DIR=/some/other/folder)
DATA_DIR = Path(os.environ.get("IBD_DATA_DIR", ROOT / "data"))

# Everything the pipeline writes goes under this folder
RESULTS_DIR = Path(os.environ.get("IBD_RESULTS_DIR", ROOT / "results"))

# --- input file names (inside DATA_DIR) ---
# Sample metadata + FFQ answers: one row per sample (CSV)
METADATA_FILE = "hmp2_metadata_2018-08-20.csv"

# Species abundance table: rows = samples, columns = species (CSV).
# The first column must hold the same sample IDs as the metadata
# column given in SAMPLE_ID_COLUMN below.
ABUNDANCE_FILE = "species_abundance_processed.csv"

# OPTIONAL raw MetaPhlAn taxonomy table (TSV, rows = taxa, columns =
# samples). If this file exists in DATA_DIR it is used instead of
# ABUNDANCE_FILE and the species are filtered by MIN_PREVALENCE.
# Set to None if you only have the processed table.
TAXONOMY_FILE = "hmp2_mgx_taxonomy.tsv"

# Only used with the raw taxonomy file
MIN_PREVALENCE = 0.10        # keep species seen in >10% of samples
PREFER_P_PROFILE = True      # if a sample has both "_profile" and "_P_profile" columns, use the P one

# --- output folders (inside RESULTS_DIR) ---
D1_DIR = RESULTS_DIR / "D1_Dietary_Analysis"
D2_DIR = RESULTS_DIR / "D2_Ghost_Biomarker"
D3_DIR = RESULTS_DIR / "D3_Food_Microbiome"
D4_DIR = RESULTS_DIR / "D4_Reproducible_Pipeline"   # shared tables, audit, workbook, run report

D1_TAB_DIR, D1_FIG_DIR = D1_DIR / "tables", D1_DIR / "figures"
D2_TAB_DIR, D2_FIG_DIR = D2_DIR / "tables", D2_DIR / "figures"
D3_TAB_DIR, D3_FIG_DIR = D3_DIR / "tables", D3_DIR / "figures"
D4_TAB_DIR = D4_DIR / "tables"

# Intermediate analysis tables written by step 01 and read by the later steps
ANALYSIS_DIR = D4_DIR / "analysis_tables"
MASLIN_DIR = D4_DIR / "maaslin2_runs"    # only used with --engine maaslin2


def make_output_dirs():
    for folder in (D1_TAB_DIR, D1_FIG_DIR, D2_TAB_DIR, D2_FIG_DIR,
                   D3_TAB_DIR, D3_FIG_DIR, D4_TAB_DIR, ANALYSIS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. METADATA COLUMN NAMES
# ======================================================================
# Left side  = the short name used inside the pipeline (don't change these)
# Right side = the column name in YOUR metadata file (change these)

SAMPLE_ID_COLUMN = "External ID"       # must match the first column of the abundance table

SAMPLE_COLUMNS = {
    # required
    "Subject":    "Participant ID",    # participant identifier (repeat samples share it)
    "diagnosis":  "diagnosis",
    "Age":        "consent_age",
    "Antibiotics": "Antibiotics",
    "Site":       "site_name",
    # optional - delete a line if your data doesn't have it
    "week_num":   "week_num",
    "sex":        "sex",
    "fecalcal":   "fecalcal",          # fecal calprotectin; without it the calprotectin part of D3 is skipped
}

# The metadata file has many data types (metagenomics, proteomics...).
# Keep only the rows where DATA_TYPE_COLUMN == DATA_TYPE_VALUE.
# Set DATA_TYPE_COLUMN to None if your file is already filtered.
DATA_TYPE_COLUMN = "data_type"
DATA_TYPE_VALUE = "metagenomics"


# ======================================================================
# 3. FFQ SCORING
# ======================================================================
# How the text answers in the FFQ columns are turned into numbers.
# Change the numbers (or the wording of the answers) to match your own
# questionnaire. The keys must match the answers in your file EXACTLY.
#
# Example: to score by approximate days in the last week instead, you
# could use 0 / 5.5 / 2.5 / 1 / 1 etc. -- whatever suits your design.
#
# If your FFQ columns are ALREADY numeric (e.g. servings per week),
# set FFQ_SCORES = None and the numbers are used as they are.

FFQ_SCORES = {
    "No, I did not consume these products in the last 7 days": 0,
    "Within the past 4 to 7 days": 1,
    "Within the past 2 to 3 days": 2,
    "Yesterday, 1 to 2 times": 3,
    "Yesterday, 3 or more times": 4,
}

# What to do with an answer that is not in FFQ_SCORES (and isn't blank):
#   "error"   -> stop and list the unrecognised answers (safest)
#   "missing" -> treat it as missing (the sample is then dropped)
FFQ_UNKNOWN_ANSWER = "error"

# Used on plot axes only
FFQ_SCORE_LABEL = "frequency score"


# ======================================================================
# 4. DIETARY ITEMS
# ======================================================================
# short name used in the pipeline  ->  FFQ column in the metadata.
# The column can be given as the full name or as the beginning of the
# name (long HMP2 column names are cut off here). The beginning must be
# unique, otherwise the check script will tell you which columns clash.
#
# Add / remove items freely. Short names: letters, numbers, underscores,
# and they must not end in "_Mean" or "_Dev".

FOODS = {
    "Fruits":          "Fruits (no juice)",
    "Vegetables":      "Vegetables (salad",
    "Beans":           "Beans (tofu",
    "Whole_Grains":    "Whole grains (wheat",
    "Starch":          "Starch (white rice",
    "Red_Meat":        "Red meat (beef",
    "Sweets":          "Sweets (pies",
    "Sugary_Drinks":   "Soft drinks, tea or coffee with sugar",
    "Processed_Meat":  "Processed meat (other red",
    "White_Meat":      "White meat (chicken",
    "Dairy":           "Dairy (milk",
    "Eggs":            "Eggs",
    "Fish":            "Fish (fish nuggets",
    "Diet_Drinks":     "Diet soft drinks",
    "Fruit_Juice":     "Fruit juice",
    "Water":           "Water",
    "Alcohol":         "Alcohol (beer",
    "Yogurt_Cultures": "Yogurt or other foods",
    "Probiotic":       "Probiotic",
    "Shellfish":       "Shellfish (shrimp",
    "Tea_Coffee_NoSugar": "Tea or coffee no sugar",
}

# Foods put into the "small" diet-adjusted models (D2 primary analysis).
# Must be keys of FOODS.
MODEL_FOODS = ["Whole_Grains", "Red_Meat", "Sugary_Drinks"]

# Extra food-score(s) added as plain covariates in the "+ probiotic"
# sensitivity model. Leave as [] to skip that model.
EXTRA_COVARIATE_FOODS = ["Probiotic"]

# Foods highlighted in D3 (the ones named in the project proposal)
PROPOSAL_FOODS = ["Whole_Grains", "Beans", "Vegetables"]

# Smaller food set used for the D1 clustering sensitivity check
# (the original 13-item list). Must be keys of FOODS.
D1_SENSITIVITY_FOODS = [
    "Fruits", "Vegetables", "Beans", "Whole_Grains", "Starch", "Red_Meat",
    "Sweets", "Sugary_Drinks", "Processed_Meat", "White_Meat", "Dairy",
    "Eggs", "Fish",
]


# ======================================================================
# 5. COVARIATES AND GROUPS
# ======================================================================
# Diagnosis groups. The FIRST one is the reference group in the models.
DIAG_LEVELS = ["nonIBD", "CD", "UC"]

# Covariates in the disease-taxon models (names = pipeline names from
# SAMPLE_COLUMNS above). Categorical covariates become 0/1 dummy
# columns; the first level (alphabetical) is the reference.
NUMERIC_COVARIATES = ["Age"]
CATEGORICAL_COVARIATES = ["Antibiotics", "Site"]

# Covariates in the fecal calprotectin models (D3). Site is left out on
# purpose here because only IBD samples are used.
CALPROTECTIN_NUMERIC_COVARIATES = ["Age"]
CALPROTECTIN_CATEGORICAL_COVARIATES = ["Antibiotics"]


# ======================================================================
# 6. STATISTICAL SETTINGS
# ======================================================================
ALPHA = 0.05                # main BH q-value threshold
ALPHA_LOOSE = 0.25          # exploratory threshold
ATTEN_THRESHOLD = 30.0      # % drop in |coef| needed to call a hit "attenuated"
# NB: output columns called q05 / q25 in the tables refer to ALPHA / ALPHA_LOOSE

SEED = 42
N_PERM = 2000               # permutations for correlation-difference tests (D1)
N_BOOT = 200                # bootstrap resamples (D1 cluster stability, D2 CIs)

# D1 diet clustering
K_CLUSTERS = 3              # number of diet clusters used in the analysis
K_RANGE = range(2, 7)       # k values compared by silhouette score
N_PERM_CLUSTER = 5000       # permutations for the cluster-vs-diagnosis test


# ======================================================================
# 7. MODEL SPECIFICATIONS
# ======================================================================
# "diet" = which foods go in, "mode":
#     "visit"    -> the food score at the sample's own visit
#     "mean_dev" -> each person's average (habitual) + the visit's
#                   deviation from it (weekly)
# The names include the number of foods so they stay correct if you
# change MODEL_FOODS / FOODS.

_n_small = len(MODEL_FOODS)
_n_all = len(FOODS)

SPECS = {
    "naive": {
        "diet": [], "mode": None, "extra": [],
        "label": "Naive\n(no diet)",
    },
    f"adj_visit_{_n_small}": {
        "diet": MODEL_FOODS, "mode": "visit", "extra": [],
        "label": f"{_n_small} foods\n(visit)",
    },
    f"adj_habitual_{_n_small}": {
        "diet": MODEL_FOODS, "mode": "mean_dev", "extra": [],
        "label": f"{_n_small} foods\n(habitual+weekly)\nPRIMARY",
    },
}

# the "+ probiotic" model is only added if it makes sense
if EXTRA_COVARIATE_FOODS and all(f in FOODS for f in EXTRA_COVARIATE_FOODS):
    SPECS[f"adj_habitual_{_n_small}_pb"] = {
        "diet": MODEL_FOODS, "mode": "mean_dev",
        "extra": list(EXTRA_COVARIATE_FOODS),
        "label": f"{_n_small} foods +\n" + "+".join(EXTRA_COVARIATE_FOODS).lower(),
    }

SPECS[f"adj_habitual_{_n_all}"] = {
    "diet": list(FOODS), "mode": "mean_dev", "extra": [],
    "label": f"{_n_all} foods\n({2 * _n_all} terms)",
}

# Model reported in the D2 registry
PRIMARY_SPEC = f"adj_habitual_{_n_small}"
# Model used for food-taxon associations in D3 (all foods together)
D3_SPEC = f"adj_habitual_{_n_all}"


# ======================================================================
# 8. HELPERS
# ======================================================================

def bh(p):
    """Benjamini-Hochberg q-values. NaNs stay NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    values = p[valid]

    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1)
    q[valid] = result
    return q


def check_config():
    """Catch typos in this file before any analysis runs.
    Returns a list of problem messages (empty list = all fine)."""
    problems = []

    # food names
    for name in FOODS:
        if not name.isidentifier():
            problems.append(f"FOODS: '{name}' is not a valid short name (use letters, numbers, underscores)")
        if name.endswith(("_Mean", "_Dev")):
            problems.append(f"FOODS: '{name}' must not end in _Mean or _Dev")

    for list_name in ("MODEL_FOODS", "PROPOSAL_FOODS", "D1_SENSITIVITY_FOODS", "EXTRA_COVARIATE_FOODS"):
        for food in globals()[list_name]:
            if food not in FOODS:
                problems.append(f"{list_name}: '{food}' is not a key of FOODS")

    overlap = set(MODEL_FOODS) & set(EXTRA_COVARIATE_FOODS)
    if overlap:
        problems.append(f"EXTRA_COVARIATE_FOODS overlaps with MODEL_FOODS: {sorted(overlap)}")

    # scoring
    if FFQ_SCORES is not None and len(FFQ_SCORES) == 0:
        problems.append("FFQ_SCORES is empty (use None if your FFQ columns are already numeric)")
    if FFQ_UNKNOWN_ANSWER not in ("error", "missing"):
        problems.append("FFQ_UNKNOWN_ANSWER must be 'error' or 'missing'")

    # columns / groups
    for needed in ("Subject", "diagnosis", "Age", "Antibiotics", "Site"):
        if needed not in SAMPLE_COLUMNS:
            problems.append(f"SAMPLE_COLUMNS is missing the required entry '{needed}'")
    for cov in NUMERIC_COVARIATES + CATEGORICAL_COVARIATES:
        if cov not in SAMPLE_COLUMNS:
            problems.append(f"covariate '{cov}' is not listed in SAMPLE_COLUMNS")
    if len(DIAG_LEVELS) < 2:
        problems.append("DIAG_LEVELS needs at least two groups")

    # specs
    for spec_name in (PRIMARY_SPEC, D3_SPEC):
        if spec_name not in SPECS:
            problems.append(f"model '{spec_name}' is not defined in SPECS")
    if not 2 <= K_CLUSTERS <= 8:
        problems.append("K_CLUSTERS must be between 2 and 8")

    return problems
