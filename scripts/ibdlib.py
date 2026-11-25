"""
Shared functions used by the pipeline scripts.

    - reading the metadata and the abundance table
    - turning FFQ answers into numbers (scoring comes from config.py)
    - CLR transform, Shannon diversity
    - building the model design matrices
    - OLS with cluster-robust standard errors, subject-level bootstrap
"""

import numpy as np
import pandas as pd
from scipy import stats

import config as C


# ----------------------------------------------------------------------
# Reading the input files
# ----------------------------------------------------------------------

def load_metadata():
    """Read the metadata, keep the right data type, one row per sample ID."""
    path = C.DATA_DIR / C.METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}\n"
            f"Put it in the data/ folder and set METADATA_FILE in config.py."
        )

    df = pd.read_csv(path, low_memory=False)

    if C.DATA_TYPE_COLUMN and C.DATA_TYPE_COLUMN in df.columns:
        df = df[df[C.DATA_TYPE_COLUMN] == C.DATA_TYPE_VALUE]

    return df.drop_duplicates(C.SAMPLE_ID_COLUMN).set_index(C.SAMPLE_ID_COLUMN)


def find_food_column(columns, wanted):
    """Find the metadata column for one food.

    An exact match wins. Otherwise we accept a column that starts with
    `wanted`, but only if exactly one column does.
    Returns (column_name, list_of_candidates).
    """
    columns = list(columns)
    if wanted in columns:
        return wanted, [wanted]

    candidates = [c for c in columns if str(c).startswith(wanted)]
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def food_columns(meta):
    """{short food name: metadata column}. Raises a clear error if a food
    can't be matched to exactly one column."""
    found, problems = {}, []

    for short, wanted in C.FOODS.items():
        col, candidates = find_food_column(meta.columns, wanted)
        if col is None and not candidates:
            problems.append(f"  {short}: no column starts with '{wanted}'")
        elif col is None:
            problems.append(f"  {short}: '{wanted}' matches {len(candidates)} columns: {candidates}")
        else:
            found[short] = col

    if problems:
        raise ValueError(
            "Could not match these FOODS entries in config.py to the metadata columns:\n"
            + "\n".join(problems)
        )
    return found


def score_answers(series):
    """Turn one FFQ column into numbers using C.FFQ_SCORES."""
    if C.FFQ_SCORES is None:
        return pd.to_numeric(series, errors="coerce")

    scored = series.map(C.FFQ_SCORES)

    unknown = series[series.notna() & scored.isna()].unique()
    if len(unknown) and C.FFQ_UNKNOWN_ANSWER == "error":
        raise ValueError(
            f"Column '{series.name}' has answers that are not in FFQ_SCORES (config.py):\n"
            + "\n".join(f"  {a!r}" for a in unknown)
        )
    return scored


def encode_foods(meta):
    """Numeric FFQ score for every food in C.FOODS (samples x foods)."""
    out = pd.DataFrame(index=meta.index)
    for short, col in food_columns(meta).items():
        out[short] = score_answers(meta[col])
    return out


def load_abundance(log=print):
    """Species abundance, samples x species.

    Uses the raw MetaPhlAn table if TAXONOMY_FILE exists, otherwise the
    processed species table (ABUNDANCE_FILE).
    Returns (table, short description of the source).
    """
    raw = C.DATA_DIR / C.TAXONOMY_FILE if C.TAXONOMY_FILE else None

    if raw is not None and raw.exists():
        tax = pd.read_csv(raw, sep="\t", index_col=0)

        # species rows only (drop higher ranks and strain-level rows)
        sp = tax[tax.index.str.contains(r"\|s__") & ~tax.index.str.contains(r"\|t__")].copy()
        sp.index = (sp.index.str.extract(r"\|s__(.*)", expand=False)
                    .str.replace(" ", "_"))

        # one profile per sample
        chosen = {}
        for col in sp.columns:
            sid = col.replace("_P_profile", "").replace("_profile", "")
            if sid not in chosen or (C.PREFER_P_PROFILE and col.endswith("_P_profile")):
                chosen[sid] = col
        sp = sp[list(chosen.values())]
        sp.columns = list(chosen)

        # total-sum scaling, then samples x species
        sp = sp.div(sp.sum(axis=0).where(lambda s: s > 0, 1), axis=1)
        ab = sp.T

        keep = (ab > 0).mean(axis=0) > C.MIN_PREVALENCE
        ab = ab.loc[:, keep]

        log(f"abundance: raw MetaPhlAn table {C.TAXONOMY_FILE} "
            f"({ab.shape[0]} samples x {ab.shape[1]} species after the "
            f">{C.MIN_PREVALENCE:.0%} prevalence filter)")
        return ab, "raw taxonomy"

    path = C.DATA_DIR / C.ABUNDANCE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Abundance file not found: {path}\n"
            f"Put it in the data/ folder and set ABUNDANCE_FILE in config.py."
        )

    ab = pd.read_csv(path, index_col=0)
    log(f"abundance: processed species table {C.ABUNDANCE_FILE} "
        f"({ab.shape[0]} samples x {ab.shape[1]} species), used as supplied "
        f"(MIN_PREVALENCE is not applied to this table)")
    return ab, "processed species table"


def clr(ab, pseudo_frac=0.5):
    # Apply CLR transformation using half the smallest non-zero abundance
    x = ab.values.astype(float)

    pc = pseudo_frac * x[x > 0].min()

    x = x + pc

    # Normalize each sample
    x = x / x.sum(axis=1, keepdims=True)

    lx = np.log(x)

    clr_values = (
        lx - lx.mean(axis=1, keepdims=True)
    )

    return pd.DataFrame(
        clr_values,
        index=ab.index,
        columns=ab.columns
    )


def shannon(ab):
    # Calculate Shannon diversity for each sample
    p = ab.div(
        ab.sum(axis=1),
        axis=0
    ).values

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):
        h = -np.nansum(
            np.where(
                p > 0,
                p * np.log(p),
                0
            ),
            axis=1
        )

    return pd.Series(
        h,
        index=ab.index,
        name="Shannon"
    )


# ----------------------------------------------------------------------
# Design matrices
# ----------------------------------------------------------------------

def dummy_columns(df, column, levels=None):
    """0/1 columns for a categorical covariate. The first level is the
    reference and gets no column. Column names are <column>_<level>."""
    if levels is None:
        levels = sorted(df[column].dropna().unique())

    out = pd.DataFrame(index=df.index)
    for level in levels[1:]:
        out[f"{column}_{str(level).replace(' ', '_')}"] = (df[column] == level).astype(float)
    return out


def covariate_block(df, numeric, categorical, levels_from=None):
    """Numeric covariates as they are, categorical ones as dummies.
    `levels_from` = table used to decide the dummy levels (default: df)."""
    ref = df if levels_from is None else levels_from
    parts = [df[numeric].astype(float)]
    for col in categorical:
        levels = sorted(ref[col].dropna().unique())
        parts.append(dummy_columns(df, col, levels))
    return pd.concat(parts, axis=1)


def diagnosis_terms():
    """Model term names for the non-reference diagnosis groups."""
    return ["diagnosis_" + g for g in C.DIAG_LEVELS[1:]]


def build_design(df, spec_name):
    """Design matrix for one model from C.SPECS."""
    spec = C.SPECS[spec_name]

    X = pd.DataFrame(index=df.index)
    X["Intercept"] = 1.0

    # diagnosis (first level in DIAG_LEVELS is the reference)
    for group in C.DIAG_LEVELS[1:]:
        X["diagnosis_" + group] = (df["diagnosis"] == group).astype(float)

    # covariates
    X = pd.concat([X, covariate_block(df, C.NUMERIC_COVARIATES, C.CATEGORICAL_COVARIATES)], axis=1)

    # diet
    foods = spec["diet"]
    if spec["mode"] == "visit":
        for food in foods:
            X[food] = df[food].astype(float)
    elif spec["mode"] == "mean_dev":
        for food in foods:
            X[food + "_Mean"] = df[food + "_Mean"]
        for food in foods:
            X[food + "_Dev"] = df[food + "_Dev"]

    for extra in spec["extra"]:
        X[extra] = df[extra].astype(float)

    return X


def add_person_mean_dev(df, foods):
    # Calculate each subject's average food intake and visit-level deviation
    df = df.copy()

    for food in foods:

        mean = (
            df.groupby("Subject")[food]
            .transform("mean")
        )

        df[food + "_Mean"] = mean
        df[food + "_Dev"] = df[food] - mean

    return df


def ols_cr1(X, Y, groups):
    # Fit OLS for multiple outcomes using the same design matrix.
    # Calculate CR1 cluster-robust standard errors.
    # P-values use a t-distribution with G-1 degrees of freedom.
    #
    # X: n x k design matrix
    # Y: n x p outcome matrix
    # groups: subject labels for each observation
    #
    # Returns coefficients, standard errors, p-values, and cluster count.

    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    n, k = X.shape

    # Identify unique clusters
    codes, ginv = np.unique(
        groups,
        return_inverse=True
    )

    G = len(codes)

    # Calculate OLS coefficients
    XtXi = np.linalg.pinv(X.T @ X)

    B = XtXi @ X.T @ Y

    # Calculate residuals
    U = Y - X @ B

    # Create a matrix identifying cluster membership
    M = np.zeros((G, n))

    M[ginv, np.arange(n)] = 1.0

    # CR1 correction factor
    c = (
        G / (G - 1)
        * (n - 1) / (n - k)
    )

    se = np.empty_like(B)

    # Calculate cluster-robust standard errors for each outcome
    for j in range(Y.shape[1]):

        S = M @ (X * U[:, [j]])

        V = (
            c
            * XtXi
            @ (S.T @ S)
            @ XtXi
        )

        se[:, j] = np.sqrt(
            np.clip(
                np.diag(V),
                0,
                None
            )
        )

    # Calculate t-statistics and p-values
    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):
        t = B / se

    p = 2 * stats.t.sf(
        np.abs(t),
        df=G - 1
    )

    return B, se, p, G


def fit_python(df, Y, spec_name):
    # Fit one model specification for all outcomes in Y
    # Return the results in a tidy table

    X = build_design(df, spec_name)

    B, se, p, G = ols_cr1(
        X.values,
        Y.values,
        df["Subject"].values
    )

    rows = []

    for i, term in enumerate(X.columns):

        # Do not include the intercept in the results
        if term == "Intercept":
            continue

        rows.append(
            pd.DataFrame({
                "spec": spec_name,
                "feature": Y.columns,
                "term": term,
                "coef": B[i],
                "se": se[i],
                "pval": p[i],
                "n": len(df),
                "clusters": G,
            })
        )

    return pd.concat(
        rows,
        ignore_index=True
    )


def cluster_bootstrap_coefs(
    df,
    Y,
    spec_names,
    terms,
    B=200,
    seed=C.SEED
):
    # Perform subject-level bootstrap for multiple model specifications.
    # All specifications use the same resampled data in each iteration.
    #
    # Returns:
    # {spec: array with shape (B, n_terms, n_outcomes)}

    rng = np.random.default_rng(seed)

    subj = df["Subject"].values
    ids = np.unique(subj)

    # Store row indices for each subject
    rows_by = {
        subject: np.where(subj == subject)[0]
        for subject in ids
    }

    # Build design matrices for all requested models
    Xs = {
        spec: build_design(df, spec)
        for spec in spec_names
    }

    # Find the positions of the requested terms
    cols = {
        spec: [
            list(Xs[spec].columns).index(term)
            for term in terms
        ]
        for spec in spec_names
    }

    # Prepare arrays for bootstrap coefficients
    out = {
        spec: np.empty(
            (B, len(terms), Y.shape[1])
        )
        for spec in spec_names
    }

    Yv = Y.values

    # Run the bootstrap
    for b in range(B):

        # Randomly select subjects with replacement
        pick = np.concatenate([
            rows_by[subject]
            for subject in rng.choice(
                ids,
                len(ids),
                replace=True
            )
        ])

        # Fit every model using the same resampled subjects
        for spec in spec_names:

            Xb = Xs[spec].values[pick]

            coef = np.linalg.lstsq(
                Xb,
                Yv[pick],
                rcond=None
            )[0]

            out[spec][b] = coef[cols[spec]]

    return out
