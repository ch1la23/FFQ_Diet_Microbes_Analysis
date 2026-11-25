# Model engines for the HMP2 diet-confounding pipeline.
#
# Both engines return the same table:
# spec, feature, term, coef, se, pval, n, clusters.
#
# python:
# Uses CLR-transformed abundance with OLS and subject-cluster-robust
# (CR1) standard errors. Requires NumPy and SciPy.
#
# maaslin2:
# Uses MaAsLin2 in R with total-sum-scaled abundance, LOG transformation,
# and a linear mixed model with a Subject random intercept.
# Requires R and Bioconductor Maaslin2.
#
# Both engines use the same model specifications but different
# statistical methods, so their results may differ.

import shutil
import subprocess

import pandas as pd

import config as C
import ibdlib as L


def run_python(df, ab, specs):
    # Run the Python engine using CLR-transformed abundance

    Y = L.clr(ab)

    results = [
        L.fit_python(df, Y, spec)
        for spec in specs
    ]

    return pd.concat(
        results,
        ignore_index=True
    )


def run_maaslin2(df, ab, specs):
    # Run MaAsLin2 for each requested model specification

    if shutil.which("Rscript") is None:
        raise RuntimeError(
            "Rscript not found. Install R and "
            "BiocManager::install('Maaslin2'), "
            "or use --engine python."
        )

    working_dir = C.MASLIN_DIR
    working_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for spec in specs:

        # Prepare metadata for the current model
        X = L.build_design(df, spec).drop(
            columns="Intercept"
        )

        X.insert(
            0,
            "Subject",
            df["Subject"].values
        )

        metadata_file = working_dir / f"{spec}_metadata.csv"
        abundance_file = working_dir / f"{spec}_abundance.csv"
        output_dir = working_dir / spec

        X.to_csv(
            metadata_file,
            index_label="Sample"
        )

        ab.to_csv(
            abundance_file,
            index_label="Sample"
        )

        # Include all model variables except Subject as fixed effects
        fixed = ",".join(
            column
            for column in X.columns
            if column != "Subject"
        )

        # Run the MaAsLin2 R script
        subprocess.run(
            [
                "Rscript",
                str(C.ROOT / "scripts" / "R" / "run_maaslin2.R"),
                str(abundance_file),
                str(metadata_file),
                str(output_dir),
                fixed,
            ],
            check=True
        )

        # Read the MaAsLin2 results
        result_file = output_dir / "all_results.tsv"

        r = pd.read_csv(
            result_file,
            sep="\t"
        )

        r["term"] = r["metadata"]

        r = r.rename(
            columns={"stderr": "se"}
        )

        r.insert(
            0,
            "spec",
            spec
        )

        r["n"] = len(df)
        r["clusters"] = df["Subject"].nunique()

        # Keep the same output columns as the Python engine
        results.append(
            r[
                [
                    "spec",
                    "feature",
                    "term",
                    "coef",
                    "se",
                    "pval",
                    "n",
                    "clusters",
                ]
            ]
        )

    return pd.concat(
        results,
        ignore_index=True
    )


ENGINES = {
    "python": run_python,
    "maaslin2": run_maaslin2,
}

