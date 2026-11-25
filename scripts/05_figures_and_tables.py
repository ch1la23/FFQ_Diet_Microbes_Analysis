
"""
Step 5: figures for D2 / D3 and the combined Excel workbook.

Reads the tables written by steps 1-4 (nothing is recomputed here).
Labels and counts in titles come from the tables and config.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # config.py
sys.path.insert(0, str(HERE))           # ibdlib.py, engines.py

import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import config as C

C.make_output_dirs()


def load_table(name):
    if name == "audit.csv":
        folder = C.D4_TAB_DIR
    elif name.startswith("d1_"):
        folder = C.D1_TAB_DIR
    elif name.startswith("d2_"):
        folder = C.D2_TAB_DIR
    elif name.startswith("d3_"):
        folder = C.D3_TAB_DIR
    else:
        raise ValueError(f"Unknown table location: {name}")

    return pd.read_csv(folder / name)


def clean_name(name):
    return name.replace("_", " ")


sens = load_table("d2_sensitivity_by_spec.csv")

# Colours used to identify different biomarker categories
colors = {
    "Not significant": "#C9CED3",
    "Persistent": "#1F77B4",
    "Ghost (diet-driven)": "#C0392B",
    "Attenuated by diet": "#E67E22",
    "Lost FDR, effect intact": "#8E44AD",
    "Emergent after adjustment": "#2CA02C"
}


# Compare disease coefficients before and after diet adjustment
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6))

comparisons = [
    (
        ax[0],
        C.PRIMARY_SPEC,
        f"Primary: + {len(C.MODEL_FOODS)} foods (habitual + weekly)"
    ),
    (
        ax[1],
        C.D3_SPEC,
        f"Sensitivity: + all {len(C.FOODS)} foods ({2 * len(C.FOODS)} terms)"
    )
]

for plot, spec, title in comparisons:

    data = load_table(f"d2_registry_{spec}.csv")

    for category, group in data.groupby("class"):

        plot.scatter(
            group.coef_naive,
            group.coef_adj,
            s=22 if category == "Not significant" else 60,
            c=colors[category],
            label=f"{category} ({len(group)})",
            edgecolor="white",
            lw=0.4,
            zorder=3 if category != "Not significant" else 2
        )

    limit = (
        np.abs(data[["coef_naive", "coef_adj"]].values).max()
        * 1.05
    )

    plot.plot(
        [-limit, limit],
        [-limit, limit],
        "k--",
        lw=0.8
    )

    plot.set_xlim(-limit, limit)
    plot.set_ylim(-limit, limit)

    plot.set_xlabel("Disease coefficient, naive model (CLR)")
    plot.set_ylabel("Disease coefficient, diet-adjusted")
    plot.set_title(title, fontweight="bold", fontsize=11)

    plot.legend(fontsize=7.5, loc="upper left")

plt.tight_layout()
plt.savefig(C.D2_FIG_DIR / "d2_coef_naive_vs_adjusted.png", dpi=170)
plt.close()


# Compare significant results in naive and adjusted models
data = load_table(f"d2_registry_{C.PRIMARY_SPEC}.csv")

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

comparisons = [
    (ax[0], "coef_naive", "q_naive", "Naive"),
    (ax[1], "coef_adj", "q_adj", "Diet-adjusted (primary)")
]

for plot, coef, q_value, title in comparisons:

    plot.scatter(
        data[coef],
        -np.log10(data[q_value]),
        s=16,
        c=np.where(
            data[q_value] < C.ALPHA,
            "#C0392B",
            "#B0B7BD"
        )
    )

    plot.axhline(
        -np.log10(C.ALPHA),
        color="k",
        ls="--",
        lw=0.8
    )

    significant = data[data[q_value] < C.ALPHA]

    plot.set_title(
        f"{title}: {len(significant)} at q < {C.ALPHA}",
        fontweight="bold"
    )

    plot.set_xlabel("Disease coefficient (CLR)")

    for _, row in significant.iterrows():

        plot.annotate(
            f"{clean_name(row.feature)} ({row.contrast})",
            (row[coef], -np.log10(row[q_value])),
            fontsize=6.5,
            xytext=(3, 3),
            textcoords="offset points"
        )

ax[0].set_ylabel(f"-log10 BH q (over {len(data)} disease tests)")

plt.tight_layout()
plt.savefig(C.D2_FIG_DIR / "d2_volcano.png", dpi=170)
plt.close()


# Show how the number of significant results changes across models
fig, ax = plt.subplots(figsize=(8, 4.2))

# labels come from the "label" entry of each model in config.SPECS
labels = {spec: C.SPECS[spec]["label"] for spec in C.SPECS if spec != "naive"}

x = np.arange(len(sens))

ax.bar(
    x - 0.2,
    [sens.naive_q05.iloc[0]] * len(sens),
    0.38,
    color="#95A5A6",
    label="naive"
)

ax.bar(
    x + 0.2,
    sens.adj_q05,
    0.38,
    color="#1F77B4",
    label="diet-adjusted"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [labels[spec] for spec in sens.spec],
    fontsize=8
)

ax.set_ylabel(f"Disease-taxon tests with q < {C.ALPHA}")
ax.legend()
ax.set_title(
    "Hit count depends on how diet is represented",
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(C.D2_FIG_DIR / "d2_hits_by_spec.png", dpi=170)
plt.close()


# Create a heatmap showing food and microbial species associations
results = load_table("d3_food_taxon_associations.csv")

top_taxa = (
    results[results.q_global < 0.25]
    .feature.value_counts()
    .head(14).index
)

top_foods = load_table("d3_food_ranking.csv").head(12).food

between = results[
    results.kind.str.startswith("between") &
    results.feature.isin(top_taxa) &
    results.food.isin(top_foods)
]

matrix = (
    between.assign(
        value=-np.log10(between.pval) * np.sign(between.coef)
    )
    .pivot(
        index="food",
        columns="feature",
        values="value"
    )
    .reindex(index=top_foods)
)

q_matrix = (
    between.pivot(
        index="food",
        columns="feature",
        values="q_global"
    )
    .reindex(index=top_foods, columns=matrix.columns)
)

annotations = q_matrix.map(
    lambda q: "**" if q < 0.05 else "*" if q < 0.25 else ""
)

fig, ax = plt.subplots(figsize=(11, 5.6))

sns.heatmap(
    matrix.rename(
        index=clean_name,
        columns=clean_name
    ),
    cmap="RdBu_r",
    center=0,
    vmin=-4,
    vmax=4,
    annot=annotations.values,
    fmt="",
    linewidths=0.5,
    linecolor="white",
    ax=ax,
    cbar_kws={
        "label": "Signed -log10 p (between-person)"
    }
)

ax.set_title(
    "Habitual food intake vs species (CLR). "
    "** global q<0.05, * q<0.25",
    fontweight="bold"
)

ax.set_xlabel("")
ax.set_ylabel("")

plt.xticks(rotation=55, ha="right", fontsize=8)
plt.tight_layout()

plt.savefig(
    C.D3_FIG_DIR / "d3_food_taxon_heatmap.png",
    dpi=170
)

plt.close()


# Plot the ranking of all foods
ranking = load_table("d3_food_ranking.csv")
n_species = results.feature.nunique()
terms_per_food = int(results.groupby("food").size().max())

fig, ax = plt.subplots(figsize=(8.5, 5.6))

y = np.arange(len(ranking))[::-1]

ax.barh(
    y,
    ranking.taxa_q25,
    color="#9DB7D5",
    label="Species at q < 0.25"
)

ax.barh(
    y,
    ranking.taxa_q05,
    color="#C0392B",
    label="Species at q < 0.05"
)

ax.set_yticks(y)

ax.set_yticklabels(
    [clean_name(food) for food in ranking.food],
    fontsize=8.5
)

ax.set_xlabel(f"Number of associated species (of {n_species})")
ax.legend(loc="lower right")

ax.set_title(
    f"Association evidence per food ({len(C.FOODS)}-food model, {terms_per_food} terms each)",
    fontweight="bold",
    fontsize=10
)

plt.tight_layout()
plt.savefig(C.D3_FIG_DIR / "d3_food_ranking.png", dpi=170)
plt.close()


# Plot food associations with fecal calprotectin (only if D3 produced them)
has_calprotectin = (C.D3_TAB_DIR / "d3_calprotectin_foods.csv").exists()

if has_calprotectin:
    cal_all = load_table("d3_calprotectin_foods.csv")
    cal = (
        cal_all
        .head(12)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    ax.errorbar(
        cal.coef,
        range(len(cal)),
        xerr=1.96 * cal.se,
        fmt="o",
        color="#34495E",
        ecolor="#95A5A6",
        capsize=3
    )

    ax.axvline(0, color="k", lw=0.8)

    ax.set_yticks(range(len(cal)))

    ax.set_yticklabels(
        [clean_name(term) for term in cal.term],
        fontsize=8.5
    )

    for i, (_, row) in enumerate(cal.iterrows()):
        ax.text(
            ax.get_xlim()[1],
            i,
            f" q={row.q:.2f}",
            va="center",
            fontsize=7.5
        )

    ax.set_xlabel(
        f"Change in log10 fecal calprotectin per 1-point {C.FFQ_SCORE_LABEL} (95% CI)"
    )

    ax.set_title(
        f"Foods vs calprotectin, IBD samples ({len(cal)} smallest p of {len(cal_all)})",
        fontweight="bold",
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig(
        C.D3_FIG_DIR / "d3_calprotectin.png",
        dpi=170,
        bbox_inches="tight"
    )

    plt.close()



# Combine all report tables into one Excel workbook
with pd.ExcelWriter(
    C.D4_DIR / "HMP2_diet_microbiome_tables.xlsx"
) as writer:

    load_table("audit.csv").to_excel(
        writer,
        sheet_name="0_Sample_audit",
        index=False
    )

    load_table("d1_food_by_diagnosis.csv").to_excel(
        writer,
        sheet_name="1_D1_Food_by_diagnosis",
        index=False
    )

    load_table("d1_food_correlations.csv").sort_values(
        "q_All"
    ).to_excel(
        writer,
        sheet_name="1b_D1_Food_correlations",
        index=False
    )

    load_table("d1_cluster_profiles.csv").to_excel(
        writer,
        sheet_name="1c_D1_Cluster_profiles",
        index=False
    )

    load_table("d2_ghost_registry_primary.csv").to_excel(
        writer,
        sheet_name="2_D2_Ghost_registry",
        index=False
    )

    sens.to_excel(
        writer,
        sheet_name="2b_D2_Sensitivity",
        index=False
    )

    load_table("d3_food_ranking.csv").to_excel(
        writer,
        sheet_name="3_D3_Food_ranking",
        index=False
    )

    results.head(300).to_excel(
        writer,
        sheet_name="3b_D3_Top_associations",
        index=False
    )

    if has_calprotectin:
        load_table("d3_calprotectin_foods.csv").to_excel(
            writer,
            sheet_name="3c_D3_Calprotectin",
            index=False
        )


print(
    "Figures:",
    sorted(p.name for p in C.D1_FIG_DIR.glob("*.png")) +
    sorted(p.name for p in C.D2_FIG_DIR.glob("*.png")) +
    sorted(p.name for p in C.D3_FIG_DIR.glob("*.png"))
)
