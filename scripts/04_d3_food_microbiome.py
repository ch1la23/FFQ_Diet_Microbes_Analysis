
"""
Step 4 / Deliverable 3: Food and Microbiome Ranking.

    - food-taxon associations from the all-foods model (config.D3_SPEC)
    - ranking of foods by how many taxa they are associated with
    - food vs Shannon diversity
    - food / taxon vs fecal calprotectin (IBD samples only)

    python scripts/04_d3_food_microbiome.py [--engine python|maaslin2]
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # config.py
sys.path.insert(0, str(HERE))           # ibdlib.py, engines.py

import argparse
import json
import numpy as np
import pandas as pd

import config as C
import ibdlib as L
import engines

C.make_output_dirs()


ap = argparse.ArgumentParser()
ap.add_argument(
    "--engine",
    default="python",
    choices=list(engines.ENGINES)
)
args = ap.parse_args()

df = pd.read_csv(C.ANALYSIS_DIR / "analysis_samples.csv", index_col=0)
ab = pd.read_csv(C.ANALYSIS_DIR / "analysis_abundance.csv", index_col=0)
res = pd.read_csv(C.D2_TAB_DIR / f"model_results_{args.engine}.csv")

foods = list(C.FOODS)
summary = {}


# Get food-related associations from the selected model
results = res[
    (res.spec == C.D3_SPEC) &
    res.term.str.endswith(("_Mean", "_Dev"))
].copy()

results["food"] = results.term.str.replace(
    "_Mean|_Dev", "", regex=True
)

results["kind"] = np.where(
    results.term.str.endswith("_Mean"),
    "between-person (habitual)",
    "within-person (weekly)"
)

# Adjust p-values across all food-taxon tests
results["q_global"] = C.bh(results.pval)

# Also adjust p-values separately for between-person and within-person tests
for kind in results.kind.unique():
    selected = results.kind == kind
    results.loc[selected, "q_family"] = C.bh(
        results.loc[selected, "pval"]
    )

results = results.sort_values("pval")

results.to_csv(
    C.D3_TAB_DIR / "d3_food_taxon_associations.csv",
    index=False
)

summary.update({
    "tests": len(results),
    "q05_global": int((results.q_global < C.ALPHA).sum()),
    "q25_global": int((results.q_global < C.ALPHA_LOOSE).sum()),
    "q05_between_family": int(
        ((results.q_family < C.ALPHA) &
         results.kind.str.startswith("between")).sum()
    ),
    "q05_within_family": int(
        ((results.q_family < C.ALPHA) &
         results.kind.str.startswith("within")).sum()
    ),
    "family_size": int(len(results) / 2)
})

print(
    f"{len(results)} tests; "
    f"q<0.05 global: {summary['q05_global']}; "
    f"q<0.25 global: {summary['q25_global']}; "
    f"per-family q<0.05 between/within: "
    f"{summary['q05_between_family']}/"
    f"{summary['q05_within_family']}"
)


# Rank foods based on their microbial associations
ranking = results.groupby("food").agg(
    taxa_q05=("q_global", lambda x: int((x < C.ALPHA).sum())),
    taxa_q25=("q_global", lambda x: int((x < C.ALPHA_LOOSE).sum())),
    taxa_p05=("pval", lambda x: int((x < C.ALPHA).sum())),
    min_p=("pval", "min")
).reset_index()

ranking["between_q25"] = (
    ranking.food.map(
        results[
            results.kind.str.startswith("between") &
            (results.q_global < C.ALPHA_LOOSE)
        ].groupby("food").size()
    ).fillna(0).astype(int)
)

ranking["within_q25"] = (
    ranking.food.map(
        results[
            results.kind.str.startswith("within") &
            (results.q_global < C.ALPHA_LOOSE)
        ].groupby("food").size()
    ).fillna(0).astype(int)
)


# Check whether food-associated microbes are linked to the IBD signature
naive = res[
    (res.spec == "naive") &
    res.term.str.startswith("diagnosis_")
].copy()

naive["q"] = C.bh(naive.pval)

wide = naive.pivot(
    index="feature",
    columns="term",
    values="coef"
)

min_p = naive.groupby("feature").pval.min()

direction = pd.Series(
    np.where(
        (wide > 0).all(axis=1),
        1,
        np.where((wide < 0).all(axis=1), -1, 0)
    ),
    index=wide.index
)

# Keep taxa with the same direction for CD and UC and nominal significance
direction = direction[
    min_p.reindex(direction.index) < C.ALPHA
]

summary.update({
    "ibd_enriched_taxa": int((direction > 0).sum()),
    "ibd_depleted_taxa": int((direction < 0).sum())
})

candidates = results[results.pval < C.ALPHA].copy()

candidates["ibd_dir"] = (
    candidates.feature.map(direction).fillna(0)
)

candidates["aligned"] = (
    np.sign(candidates.coef) == -candidates.ibd_dir
)

aligned = candidates[
    candidates.ibd_dir != 0
].groupby("food").agg(
    directional_taxa=("aligned", "size"),
    health_aligned=("aligned", "sum")
).reset_index()

ranking = ranking.merge(
    aligned,
    on="food",
    how="left"
).fillna({
    "directional_taxa": 0,
    "health_aligned": 0
})

ranking["alignment_share"] = np.where(
    ranking.directional_taxa > 0,
    ranking.health_aligned /
    ranking.directional_taxa.replace(0, np.nan),
    np.nan
)

ranking = ranking.sort_values(
    ["taxa_q05", "taxa_q25", "taxa_p05"],
    ascending=False
).reset_index(drop=True)

ranking.insert(0, "rank", ranking.index + 1)

ranking.to_csv(
    C.D3_TAB_DIR / "d3_food_ranking.csv",
    index=False
)

print(ranking.head(10).round(3).to_string(index=False))


# Save the strongest associations for the proposed foods
spotlight = results[
    results.food.isin(C.PROPOSAL_FOODS)
].sort_values("pval").groupby(
    ["food", "kind"]
).head(3)

spotlight.to_csv(
    C.D3_TAB_DIR / "d3_proposal_foods_top_taxa.csv",
    index=False
)

summary["proposal_foods"] = {}

for food in C.PROPOSAL_FOODS:
    food_result = ranking[ranking.food == food].iloc[0]

    summary["proposal_foods"][food] = {
        "q25": int(food_result.taxa_q25),
        "q05": int(food_result.taxa_q05),
        "p05": int(food_result.taxa_p05),
        "rank": int(food_result["rank"])
    }


# Check how food intake relates to Shannon diversity
X = L.build_design(df, C.D3_SPEC)

shannon = L.shannon(ab).reindex(df.index)

B, se, p, G = L.ols_cr1(
    X.values,
    shannon.values[:, None],
    df.Subject.values
)

shannon_results = pd.DataFrame({
    "term": X.columns,
    "coef": B[:, 0],
    "se": se[:, 0],
    "pval": p[:, 0]
})

shannon_results = shannon_results[
    shannon_results.term.str.endswith(("_Mean", "_Dev"))
].copy()

shannon_results["q"] = C.bh(shannon_results.pval)

shannon_results = shannon_results.sort_values("pval")

shannon_results.to_csv(
    C.D3_TAB_DIR / "d3_shannon_by_food.csv",
    index=False
)

# Keep diagnosis coefficients for the summary
diagnosis_coef = dict(zip(X.columns, B[:, 0]))
diagnosis_p = dict(zip(X.columns, p[:, 0]))

summary["shannon_q05"] = int(
    (shannon_results.q < C.ALPHA).sum()
)

summary["shannon_p05"] = int(
    (shannon_results.pval < C.ALPHA).sum()
)

summary["shannon_top"] = (
    shannon_results.iloc[0][
        ["term", "coef", "pval", "q"]
    ].to_dict()
)

summary["shannon_diag_coef"] = {
    term: [
        float(diagnosis_coef[term]),
        float(diagnosis_p[term])
    ]
    for term in L.diagnosis_terms()
}

print(
    "Shannon: q<0.05 terms:",
    summary["shannon_q05"],
    "| top:",
    summary["shannon_top"]
)


# Fecal calprotectin analysis (skipped if the data has no calprotectin column)
if "fecalcal" in df.columns and df.fecalcal.notna().any():

    ibd_groups = C.DIAG_LEVELS[1:]      # everyone except the reference (non-disease) group

    ibd = df[
        df.diagnosis.isin(ibd_groups) &
        df.fecalcal.notna()
    ].copy()

    ibd["logcal"] = np.log10(
        ibd.fecalcal.astype(float)
    )

    summary.update({
        "cal_samples": len(ibd),
        "cal_participants": int(ibd.Subject.nunique()),
        **{
            f"cal_n_{group}": int((ibd.diagnosis == group).sum())
            for group in ibd_groups
        }
    })

    # Prepare the basic model variables
    # (first disease group is the reference, the others get a 0/1 column)
    base = pd.DataFrame({"Intercept": 1.0}, index=ibd.index)

    for group in ibd_groups[1:]:
        base["diagnosis_" + group] = (ibd.diagnosis == group).astype(float)

    base = pd.concat([
        base,
        L.covariate_block(
            ibd,
            C.CALPROTECTIN_NUMERIC_COVARIATES,
            C.CALPROTECTIN_CATEGORICAL_COVARIATES,
            levels_from=df,
        ),
    ], axis=1)


    # Check the association between each food and calprotectin
    rows = []

    for food in foods:

        X = base.assign(
            **{
                food + "_Mean": ibd[food + "_Mean"],
                food + "_Dev": ibd[food + "_Dev"]
            }
        )

        if np.linalg.matrix_rank(X.values) < X.shape[1]:
            continue

        B, se, p, G = L.ols_cr1(
            X.values,
            ibd.logcal.values[:, None],
            ibd.Subject.values
        )

        for i, term in enumerate(X.columns):
            if term.startswith(food):
                rows.append({
                    "food": food,
                    "term": term,
                    "coef": B[i, 0],
                    "se": se[i, 0],
                    "pval": p[i, 0]
                })


    cal_results = pd.DataFrame(rows)

    cal_results["q"] = C.bh(cal_results.pval)
    cal_results = cal_results.sort_values("pval")

    cal_results.to_csv(
        C.D3_TAB_DIR / "d3_calprotectin_foods.csv",
        index=False
    )


    # Check the association between microbial species and calprotectin
    Y = L.clr(ab).loc[ibd.index]

    X = base.assign(logcal=ibd.logcal)

    B, se, p, G = L.ols_cr1(
        X.values,
        Y.values,
        ibd.Subject.values
    )

    taxa_results = pd.DataFrame({
        "feature": Y.columns,
        "coef": B[-1],
        "se": se[-1],
        "pval": p[-1]
    })

    taxa_results["q"] = C.bh(taxa_results.pval)
    taxa_results = taxa_results.sort_values("pval")

    taxa_results.to_csv(
        C.D3_TAB_DIR / "d3_calprotectin_taxa.csv",
        index=False
    )


    # Collect the calprotectin results
    summary.update({
        "cal_food_tests": len(cal_results),
        "cal_food_q05": int((cal_results.q < C.ALPHA).sum()),
        "cal_food_p05": int((cal_results.pval < C.ALPHA).sum()),
        "cal_food_min_q": float(cal_results.q.min()),
        "cal_taxa_tests": len(taxa_results),
        "cal_taxa_q05": int((taxa_results.q < C.ALPHA).sum()),
        "cal_taxa_p05": int((taxa_results.pval < C.ALPHA).sum()),
        "cal_taxa_min_q": float(taxa_results.q.min())
    })

    print({
        key: value
        for key, value in summary.items()
        if key.startswith("cal")
    })



else:
    print("No fecal calprotectin data; skipping the calprotectin part.")


# Save the complete summary
with open(C.D3_DIR / "d3_summary.json", "w") as file:
    json.dump(summary, file, indent=2, default=str)


print("\nTop associations:")

print(
    results.head(12)[
        [
            "food",
            "kind",
            "feature",
            "coef",
            "pval",
            "q_global",
            "q_family"
        ]
    ].round(4).to_string(index=False)
)
