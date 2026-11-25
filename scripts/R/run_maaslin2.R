
# Run MaAsLin2 using the abundance, metadata and model settings
# passed from engines.py.
# The design columns are already prepared in Python, so both
# engines use the same fixed effects. Subject is a random effect.

suppressMessages(library(Maaslin2))

args <- commandArgs(trailingOnly = TRUE)

# Read input files
abundance <- read.csv(args[1], row.names = 1, check.names = FALSE)
metadata <- read.csv(args[2], row.names = 1, check.names = FALSE)

# Keep only samples present in both files
common_samples <- intersect(rownames(abundance), rownames(metadata))

abundance <- abundance[common_samples, ]
metadata <- metadata[common_samples, ]

# Get the fixed effects from the command-line arguments
fixed_effects <- strsplit(args[4], ",")[[1]]

# Run MaAsLin2
Maaslin2(
  input_data = abundance,
  input_metadata = metadata,
  output = args[3],
  fixed_effects = fixed_effects,
  random_effects = "Subject",
  normalization = "NONE",
  transform = "LOG",
  analysis_method = "LM",
  min_prevalence = 0,
  correction = "BH",
  max_significance = 1,
  cores = 1,
  plot_heatmap = FALSE,
  plot_scatter = FALSE
)

