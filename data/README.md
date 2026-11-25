# Input data

Put the input files in this folder and list their names in `config.py`
(section 1). Nothing is read from anywhere else.

| File (default name) | What it is | Required |
|---|---|---|
| `hmp2_metadata_2018-08-20.csv` | HMP2/IBDMDB sample metadata including the FFQ answers. One row per sample, a sample ID column (`External ID`), participant ID, diagnosis, age, antibiotics, site, FFQ columns, fecal calprotectin. | yes |
| `species_abundance_processed.csv` | Species abundance, rows = samples, columns = species. First column = sample ID (same IDs as the metadata). Values are relative abundances; rows are re-normalised in step 1 anyway. | yes (unless you use the raw taxonomy file) |
| `hmp2_mgx_taxonomy.tsv` | Raw MetaPhlAn taxonomy table (rows = taxa, columns = samples). Optional: if present it is used instead of the processed table and species are filtered by `MIN_PREVALENCE`. | no |

The HMP2 files are public, from https://ibdmdb.org (Lloyd-Price et al., 2019, *Nature*).
The processed species table shipped here has already been filtered to species
present in >10% of samples (124 species); `MIN_PREVALENCE` is not applied to it.

To use your own cohort, keep the same layout (one row per sample, same sample
IDs in both files) and adjust the column names in `config.py` (section 2).
Run `python scripts/00_check_inputs.py` to see whether everything matches.
