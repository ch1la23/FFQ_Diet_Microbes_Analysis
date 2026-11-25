# HMP2 / IBDMDB: does diet confound IBD microbiome signals?

Code for the project *How Dietary Patterns Shape Gut Microbiome Signals in IBD*
(HMP2/IBDMDB, food frequency questionnaire + metagenomic species profiles,
105 participants: 49 CD, 30 UC, 26 non-IBD, 1,199 samples).

| Deliverable | What it does | Script |
|---|---|---|
| D1 | Dietary pattern atlas: food differences by diagnosis, food-food correlations, diet clusters | `scripts/02_d1_dietary_atlas.py` |
| D2 | Ghost biomarker registry: disease-taxon associations with and without diet adjustment | `scripts/03_d2_ghost_biomarkers.py` |
| D3 | Food and microbiome ranking, Shannon diversity, fecal calprotectin | `scripts/04_d3_food_microbiome.py` |
| D4 | Reproducible pipeline: runs everything, audit, Excel workbook, run report | `run_pipeline.py` |

## Folder layout

```
config.py            <- all settings (edit this one)
run_pipeline.py      <- runs steps 00-05 and writes the run report
requirements.txt     <- Python package versions
data/                <- input files (see data/README.md)
scripts/
    00_check_inputs.py        checks data + config fit together
    01_preprocess.py          builds the analysis tables and the sample audit
    02_d1_dietary_atlas.py
    03_d2_ghost_biomarkers.py
    04_d3_food_microbiome.py
    05_figures_and_tables.py  figures + Excel workbook
    ibdlib.py, engines.py     shared functions (loading, scoring, models)
    R/run_maaslin2.R          only used with --engine maaslin2
    tests/test_smoke.py
results/
    D1_Dietary_Analysis/  D2_Ghost_Biomarker/  D3_Food_Microbiome/    tables/ figures/ summary json
    D4_Reproducible_Pipeline/   audit, analysis tables, Excel workbook, run_report.json
```

## Running it

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt

python scripts/00_check_inputs.py     # optional, run_pipeline.py does it too
python run_pipeline.py                # about 1 minute
python -m pytest scripts/tests        # optional
```

Python 3.10+ is needed (results here were produced with 3.13).

Useful options: `--skip 02 05` skips steps, `--skip-check` skips the input check,
`--engine maaslin2` uses MaAsLin2 instead of the Python models (see below).
Environment variables `IBD_DATA_DIR` and `IBD_RESULTS_DIR` change the data and
results folders without editing anything.

## Using your own data or settings (everything is in `config.py`)

1. **Input files** – copy them into `data/` and set `METADATA_FILE`,
   `ABUNDANCE_FILE` (and optionally `TAXONOMY_FILE`).
2. **Metadata columns** – `SAMPLE_COLUMNS` maps the pipeline's names
   (Subject, diagnosis, Age, Antibiotics, Site, ...) to the column names in your
   file. `SAMPLE_ID_COLUMN` is the sample ID column. `DATA_TYPE_COLUMN/VALUE`
   filter the HMP2 file to metagenomics rows (set the column to `None` if your
   file is already filtered).
3. **FFQ scoring** – `FFQ_SCORES` maps each answer text to a number. The text
   has to match your file exactly. Change the numbers for a different scoring
   scheme (e.g. 0 / 1 / 3 / 5 / 7 for days per week), or write your own
   questionnaire's answers. If your FFQ columns are already numeric (servings per
   week etc.) set `FFQ_SCORES = None`. Unrecognised answers stop the run with a
   list of them (`FFQ_UNKNOWN_ANSWER = "error"`), so nothing is dropped silently.
4. **Dietary items** – `FOODS` is `short name -> FFQ column` (full column name or
   a unique beginning of it). Add or remove items freely. Then update the lists
   that refer to them: `MODEL_FOODS` (foods in the small adjusted model, D2
   primary), `EXTRA_COVARIATE_FOODS`, `PROPOSAL_FOODS` (highlighted in D3),
   `D1_SENSITIVITY_FOODS`. Model names, figure labels and titles update by
   themselves (e.g. `adj_habitual_3` becomes `adj_habitual_4` if you use 4 foods).
5. **Groups and covariates** – `DIAG_LEVELS` (first = reference group),
   `NUMERIC_COVARIATES`, `CATEGORICAL_COVARIATES`, and the calprotectin covariates.
6. **Statistics** – `ALPHA`, `ALPHA_LOOSE`, `ATTEN_THRESHOLD`, `SEED`, `N_PERM`,
   `N_BOOT`, `K_CLUSTERS`, `K_RANGE`, `N_PERM_CLUSTER`.

After editing, run `python scripts/00_check_inputs.py`. It tells you which
column, answer or food does not match before the analysis starts.

## Outputs

Each deliverable folder has `tables/`, `figures/` and a `dN_summary.json`.
`results/D4_Reproducible_Pipeline/` has `tables/audit.csv` (how many samples and
participants each filter keeps), `analysis_tables/` (the cleaned inputs for all
later steps), `HMP2_diet_microbiome_tables.xlsx` (main tables in one workbook) and
`run_report.json` (see below). The files in `results/` were generated with the
default `config.py` and the Python engine.

## Methods in brief

- **Diet scores**: FFQ answers are scored with `FFQ_SCORES`. Each person's mean
  score across visits is the *habitual* intake; the visit's difference from that
  mean is the *weekly* deviation.
- **Microbiome**: species relative abundance, CLR transform (pseudocount = half
  the smallest non-zero value).
- **Models (Python engine)**: OLS on CLR values with participant-clustered (CR1)
  standard errors, BH correction. Specifications are in `config.SPECS`: naive,
  diet at the visit, habitual + weekly (primary), habitual + weekly + probiotic,
  all foods.
- **Ghost biomarker classes (D2)**: a naive hit counts as *Ghost* only if it
  loses significance after adjustment, its |coefficient| drops by at least
  `ATTEN_THRESHOLD` %, and the bootstrap CI of the drop is above zero.
- **MaAsLin2 engine** (optional): same design matrices, TSS + LOG transform,
  random intercept for participant. Needs R with the `Maaslin2` Bioconductor
  package. The Python and MaAsLin2 engines use different statistics, so their
  numbers will not match exactly.

## Reproducibility checklist

What is already fixed:
- All paths are relative to the project folder; nothing is hard-coded to a machine.
- Random seed (`SEED`) is used for every permutation, bootstrap and k-means run.
- `run_report.json` stores Python and package versions, a SHA-256 checksum of
  each input file and of `config.py`, the foods/scoring used, and step timings.
  If someone gets different numbers, compare this file with yours first.
- `requirements.txt` pins the versions used.

What else you should keep in mind:
- **Same inputs**: compare your input checksums with `run_report.json`. HMP2
  metadata is updated from time to time; the file used here is dated 2018-08-20.
- **Package versions**: k-means, the PCA and some numerical routines can differ
  slightly between scikit-learn / NumPy versions or CPU libraries. Expect tiny
  floating point differences (around 1e-9) in some tables even with the same
  seed; the cluster labels could in rare cases be numbered differently.
- **Reference levels**: categorical covariates use the alphabetical first level
  as reference (`Antibiotics = No`, first site). A different level set changes
  the coefficient names.
- **FFQ answer wording** has to match exactly (including punctuation).
- **Processed vs raw abundance**: the shipped processed table is already
  prevalence-filtered; a raw MetaPhlAn table gives a different species set.
- **R/MaAsLin2** versions matter for `--engine maaslin2`; record `sessionInfo()`
  if you use it. That path was not run when this repository was assembled
  (no R available), so check it on your machine.
- **Data access**: the HMP2 data are public but cite the original paper; do not
  upload other cohorts' participant-level data to a public repo.

## Notes and limits

This is an observational analysis; results do not show causal effects. FFQ
answers are self-reported and short-term (last 7 days), and several samples per
participant are correlated, which is why clustered standard errors are used.
With 105 participants the power to find food-level differences is low. See
`results/*/d*_summary.json` for the numbers.
