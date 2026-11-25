
"""
Step 3 / Deliverable 2: Ghost Biomarker Registry.

Fits the disease-taxon models listed in config.SPECS (naive and
diet-adjusted), compares the disease coefficients, and classifies each
disease-taxon pair (Persistent / Attenuated / Ghost / ...).

    python scripts/03_d2_ghost_biomarkers.py [--engine python|maaslin2]
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

# Run the selected analysis model
res = engines.ENGINES[args.engine](df, ab, list(C.SPECS))

res.to_csv(
    C.D2_TAB_DIR / f"model_results_{args.engine}.csv",
    index=False
)

print(
    f"engine={args.engine}: {res.spec.nunique()} specifications, "
    f"{len(res)} coefficients; n={df.shape[0]}, "
    f"participants={df.Subject.nunique()}"
)


def diag(spec):
    # Get diagnosis-related results for a particular model
    d = res[
        (res.spec == spec) &
        res.term.str.startswith("diagnosis_")
    ].copy()

    d["contrast"] = d.term.str.replace("diagnosis_", "")
    d["q"] = C.bh(d.pval)

    return d.set_index(["feature", "contrast"])


def classify(r):
    naive_sig = r.q_naive < C.ALPHA
    adjusted_sig = r.q_adj < C.ALPHA
    strong_reduction = r.atten_pct >= C.ATTEN_THRESHOLD

    if naive_sig and not adjusted_sig and strong_reduction and r.delta_ci_lo > 0:
        return "Ghost (diet-driven)"

    if naive_sig and strong_reduction:
        return "Attenuated by diet"

    if naive_sig and not adjusted_sig:
        return "Lost FDR, effect intact"

    if naive_sig:
        return "Persistent"

    if adjusted_sig:
        return "Emergent after adjustment"

    return "Not significant"


# Bootstrap the change in effect sizes between naive and adjusted models
adj_specs = [s for s in C.SPECS if s != "naive"]
terms = L.diagnosis_terms()

Y = L.clr(ab)

boot = L.cluster_bootstrap_coefs(
    df,
    Y,
    ["naive"] + adj_specs,
    terms,
    B=C.N_BOOT
)

features = list(Y.columns)
registries = {}
sensitivity = []

for spec in adj_specs:

    naive = diag("naive")
    adjusted = diag(spec)

    # Combine results from naive and adjusted models
    m = naive[
        ["coef", "se", "pval", "q"]
    ].join(
        adjusted[["coef", "se", "pval", "q"]],
        lsuffix="_naive",
        rsuffix="_adj"
    ).reset_index()

    # Calculate how much the effect size changes after diet adjustment
    m["atten_pct"] = np.where(
        m.coef_naive.abs() > 1e-12,
        100 * (1 - m.coef_adj.abs() / m.coef_naive.abs()),
        np.nan
    )

    m["sign_flip"] = (
        np.sign(m.coef_naive) != np.sign(m.coef_adj)
    )

    lower_ci = []
    upper_ci = []

    for _, row in m.iterrows():

        term_index = terms.index("diagnosis_" + row.contrast)
        feature_index = features.index(row.feature)

        difference = (
            np.abs(boot["naive"][:, term_index, feature_index])
            - np.abs(boot[spec][:, term_index, feature_index])
        )

        lower_ci.append(np.percentile(difference, 2.5))
        upper_ci.append(np.percentile(difference, 97.5))

    m["delta_ci_lo"] = lower_ci
    m["delta_ci_hi"] = upper_ci

    # Classify each biomarker based on its results
    m["class"] = m.apply(classify, axis=1)

    m["naive_hit_q25"] = m.q_naive < C.ALPHA_LOOSE
    m["adj_hit_q25"] = m.q_adj < C.ALPHA_LOOSE

    registries[spec] = m

    hits = m[m.q_naive < C.ALPHA]

    diet_columns = [
        column
        for column in L.build_design(df, spec).columns
        if column in set(
            sum(
                [
                    [food, food + "_Mean", food + "_Dev"]
                    for food in C.SPECS[spec]["diet"]
                ],
                []
            )
        )
    ]

    sensitivity.append({
        "spec": spec,
        "diet_terms": len(diet_columns),
        "tests": len(m),
        "naive_q05": int((m.q_naive < C.ALPHA).sum()),
        "adj_q05": int((m.q_adj < C.ALPHA).sum()),
        "persistent": int((m["class"] == "Persistent").sum()),
        "attenuated": int((m["class"] == "Attenuated by diet").sum()),
        "lost_effect_intact": int(
            (m["class"] == "Lost FDR, effect intact").sum()
        ),
        "ghost": int(
            (m["class"] == "Ghost (diet-driven)").sum()
        ),
        "emergent": int(
            (m["class"] == "Emergent after adjustment").sum()
        ),
        "naive_q25": int(m.naive_hit_q25.sum()),
        "adj_q25": int(m.adj_hit_q25.sum()),
        "median_atten_among_naive_hits": (
            float(hits.atten_pct.median()) if len(hits) else np.nan
        ),
        "median_atten_all": float(m.atten_pct.median()),
        "sign_flips_among_hits": int(hits.sign_flip.sum()),
        "coef_corr": float(
            np.corrcoef(m.coef_naive, m.coef_adj)[0, 1]
        )
    })


sensitivity = pd.DataFrame(sensitivity)

sensitivity.to_csv(
    C.D2_TAB_DIR / "d2_sensitivity_by_spec.csv",
    index=False
)

# Save the main registry and results for each specification
registry = registries[C.PRIMARY_SPEC].sort_values(
    ["class", "q_naive"]
)

registry.to_csv(
    C.D2_TAB_DIR / "d2_ghost_registry_primary.csv",
    index=False
)

for spec, results in registries.items():
    results.to_csv(
        C.D2_TAB_DIR / f"d2_registry_{spec}.csv",
        index=False
    )


print(sensitivity.round(3).to_string(index=False))

significant = registry[
    registry["class"] != "Not significant"
]

print(
    significant[
        [
            "feature",
            "contrast",
            "coef_naive",
            "q_naive",
            "coef_adj",
            "q_adj",
            "atten_pct",
            "class"
        ]
    ].round(3).to_string(index=False)
)

# Save a summary of the primary analysis
primary = sensitivity[
    sensitivity.spec == C.PRIMARY_SPEC
].iloc[0].to_dict()

primary["class_counts"] = registry["class"].value_counts().to_dict()
primary["engine"] = args.engine

with open(C.D2_DIR / "d2_summary.json", "w") as file:
    json.dump(primary, file, indent=2, default=str)
