"""
Step 2 / Deliverable 1: Dietary Pattern Atlas.

    - food frequency differences between diagnosis groups (Kruskal-Wallis)
    - food-food correlations, overall and per group
    - k-means diet clusters and their link to diagnosis

Everything that can be changed (foods, groups, number of clusters,
permutations) is set in config.py.
"""

import sys
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # config.py
sys.path.insert(0, str(HERE))           # ibdlib.py, engines.py

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config as C

C.make_output_dirs()


# ---------------------------------------------------------------------
# Load data and calculate participant-level mean food frequencies
# ---------------------------------------------------------------------

samples = pd.read_csv(C.ANALYSIS_DIR / "analysis_samples.csv", index_col=0)
foods = list(C.FOODS)

# disease groups first, reference group last (CD, UC, nonIBD by default)
diagnosis_order = C.DIAG_LEVELS[1:] + C.DIAG_LEVELS[:1]

SCORE_MIN = min(C.FFQ_SCORES.values()) if C.FFQ_SCORES else None
SCORE_MAX = max(C.FFQ_SCORES.values()) if C.FFQ_SCORES else None

participants = samples.groupby("Subject").agg(
    diagnosis=("diagnosis", "first"),
    **{food: (food, "mean") for food in foods},
)


def display_food_name(food):
    """Make food labels easier to read in figures."""
    return food.replace("_", " ").replace(
        "Tea Coffee NoSugar",
        "Tea/coffee (no sugar)",
    )


summary = {
    "n_participants": len(participants),
    "n_by_group": participants["diagnosis"].value_counts().to_dict(),
}


# ---------------------------------------------------------------------
# 1. Compare food frequencies across diagnosis groups
# ---------------------------------------------------------------------

food_test_rows = []

for food in foods:
    group_values = [
        participants.loc[participants["diagnosis"] == diagnosis, food]
        for diagnosis in diagnosis_order
    ]

    statistic, p_value = stats.kruskal(*group_values)

    row = {
        "food": food,
        **{
            f"mean_{diagnosis}": group_values[index].mean()
            for index, diagnosis in enumerate(diagnosis_order)
        },
        "H": statistic,
        "epsilon_sq": statistic / (len(participants) - 1),
        "p": p_value,
    }
    food_test_rows.append(row)

food_tests = (
    pd.DataFrame(food_test_rows)
    .sort_values("p")
    .reset_index(drop=True)
)
food_tests["q_BH"] = C.bh(food_tests["p"])
food_tests.to_csv(C.D1_TAB_DIR / "d1_food_by_diagnosis.csv", index=False)

summary.update(
    food_tests=len(food_tests),
    food_p_lt_0_05=int((food_tests["p"] < 0.05).sum()),
    food_q_lt_0_05=int((food_tests["q_BH"] < 0.05).sum()),
    smallest_food_p=[
        food_tests.loc[0, "food"],
        float(food_tests.loc[0, "p"]),
    ],
    smallest_food_q=float(food_tests["q_BH"].min()),
)

print(food_tests.round(3).head(8).to_string(index=False))

# Plot mean food frequency and the Kruskal-Wallis test statistics.
food_order = (
    food_tests.sort_values("H", ascending=False)["food"].tolist()
)
mean_frequency = (
    participants.groupby("diagnosis")[foods]
    .mean()
    .loc[diagnosis_order, food_order]
)
p_values = food_tests.set_index("food")["p"]


def significance_mark(p_value):
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "~"
    return ""


annotations = mean_frequency.copy().astype(object)

for food in food_order:
    for diagnosis in diagnosis_order:
        value = mean_frequency.loc[diagnosis, food]
        annotations.loc[diagnosis, food] = (
            f"{value:.2f}{significance_mark(p_values[food])}"
        )

fig, axes = plt.subplots(
    2,
    1,
    figsize=(15, 8.5),
    gridspec_kw={"height_ratios": [1.5, 1]},
)

sns.heatmap(
    mean_frequency,
    annot=annotations,
    fmt="",
    cmap="YlGnBu",
    linewidths=0.6,
    linecolor="white",
    ax=axes[0],
    cbar_kws={
        "label": (f"Mean {C.FFQ_SCORE_LABEL} ({SCORE_MIN}–{SCORE_MAX})"
                  if SCORE_MIN is not None else f"Mean {C.FFQ_SCORE_LABEL}"),
        "shrink": 0.8,
    },
    annot_kws={"fontsize": 8},
)

axes[0].set_title(
    "Mean food frequency by diagnosis "
    f"(n = {len(participants)} participants; "
    "* raw p < 0.05, ~ raw p < 0.10)",
    fontweight="bold",
)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_xticklabels(
    [display_food_name(food) for food in food_order],
    rotation=40,
    ha="right",
)

ordered_tests = food_tests.set_index("food").loc[food_order]
bar_colors = [
    "#C0392B" if p < 0.05 else "#E67E22" if p < 0.10 else "#95A5A6"
    for p in ordered_tests["p"]
]

axes[1].bar(range(len(food_order)), ordered_tests["H"], color=bar_colors)
axes[1].set_xticks(range(len(food_order)))
axes[1].set_xticklabels(
    [display_food_name(food) for food in food_order],
    rotation=40,
    ha="right",
)
axes[1].set_ylabel("Kruskal-Wallis H")
axes[1].set_title(
    "Separation by diagnosis (colour shows raw p-value band; "
    f"{summary['food_q_lt_0_05']} of {len(food_tests)} items pass BH q < 0.05)",
    fontweight="bold",
)

plt.tight_layout()
plt.savefig(C.D1_FIG_DIR / "d1_food_by_diagnosis.png", dpi=170)
plt.close()


# ---------------------------------------------------------------------
# 2. Food-food correlations and between-group comparisons
# ---------------------------------------------------------------------

groups = {
    "All": participants,
    **{
        diagnosis: participants[participants["diagnosis"] == diagnosis]
        for diagnosis in diagnosis_order
    },
}

food_pairs = pd.DataFrame(
    itertools.combinations(foods, 2),
    columns=["food_1", "food_2"],
)


def ranked_correlation_matrix(data):
    """Calculate a Spearman correlation matrix using ranked values."""
    ranked_values = np.apply_along_axis(
        stats.rankdata,
        axis=0,
        arr=data[foods].to_numpy(),
    )
    return np.corrcoef(ranked_values, rowvar=False)


# Calculate the correlation and adjusted p-value for each pair in each group.
for group_name, group_data in groups.items():
    correlations = []
    p_values = []

    for food_1, food_2 in zip(food_pairs["food_1"], food_pairs["food_2"]):
        correlation, p_value = stats.spearmanr(
            group_data[food_1],
            group_data[food_2],
        )
        correlations.append(correlation)
        p_values.append(p_value)

    food_pairs[f"rho_{group_name}"] = correlations
    food_pairs[f"p_{group_name}"] = p_values
    food_pairs[f"q_{group_name}"] = C.bh(
        np.nan_to_num(p_values, nan=1.0)
    )

# Compare correlation strengths between diagnosis groups using permutations.
rng = np.random.default_rng(C.SEED)
upper_triangle = np.triu_indices(len(foods), k=1)

group_comparisons = list(itertools.combinations(diagnosis_order, 2))

for group_1, group_2 in group_comparisons:
    values_1 = groups[group_1][foods].to_numpy()
    values_2 = groups[group_2][foods].to_numpy()

    correlation_1 = ranked_correlation_matrix(groups[group_1])
    correlation_2 = ranked_correlation_matrix(groups[group_2])

    observed_difference = (
        np.arctanh(
            np.clip(
                np.nan_to_num(correlation_1[upper_triangle]),
                -0.999,
                0.999,
            )
        )
        - np.arctanh(
            np.clip(
                np.nan_to_num(correlation_2[upper_triangle]),
                -0.999,
                0.999,
            )
        )
    )

    pooled_values = np.vstack([values_1, values_2])
    group_1_size = len(values_1)
    exceed_count = np.zeros(len(observed_difference))

    for _ in range(C.N_PERM):
        shuffled_indices = rng.permutation(len(pooled_values))
        permuted_1 = pooled_values[shuffled_indices[:group_1_size]]
        permuted_2 = pooled_values[shuffled_indices[group_1_size:]]

        ranked_1 = np.apply_along_axis(
            stats.rankdata, 0, permuted_1
        )
        ranked_2 = np.apply_along_axis(
            stats.rankdata, 0, permuted_2
        )

        permuted_corr_1 = np.corrcoef(ranked_1, rowvar=False)[
            upper_triangle
        ]
        permuted_corr_2 = np.corrcoef(ranked_2, rowvar=False)[
            upper_triangle
        ]

        permuted_difference = (
            np.arctanh(
                np.clip(
                    np.nan_to_num(permuted_corr_1),
                    -0.999,
                    0.999,
                )
            )
            - np.arctanh(
                np.clip(
                    np.nan_to_num(permuted_corr_2),
                    -0.999,
                    0.999,
                )
            )
        )

        exceed_count += (
            np.abs(permuted_difference)
            >= np.abs(observed_difference) - 1e-12
        )

    comparison_name = f"{group_1}_vs_{group_2}"
    food_pairs[f"p_diff_{comparison_name}"] = (
        exceed_count + 1
    ) / (C.N_PERM + 1)

# Adjust the p-values across all pairwise group comparisons.
difference_p_columns = [
    column
    for column in food_pairs.columns
    if column.startswith("p_diff")
]
all_difference_q = C.bh(
    food_pairs[difference_p_columns].to_numpy().ravel(order="F")
)

for index, p_column in enumerate(difference_p_columns):
    q_column = "q_" + p_column[2:]
    start = index * len(food_pairs)
    end = start + len(food_pairs)
    food_pairs[q_column] = all_difference_q[start:end]

food_pairs.to_csv(C.D1_TAB_DIR / "d1_food_correlations.csv", index=False)

top_pairs = food_pairs.loc[
    food_pairs["rho_All"].abs().sort_values(ascending=False).index
]

difference_q_columns = [
    column
    for column in food_pairs.columns
    if column.startswith("q_diff")
]

summary.update(
    pairs_total=len(food_pairs),
    pairs_q05_all=int((food_pairs["q_All"] < 0.05).sum()),
    **{
        f"pairs_q05_{group}": int((food_pairs[f"q_{group}"] < 0.05).sum())
        for group in diagnosis_order
    },
    diff_tests=len(difference_p_columns) * len(food_pairs),
    diff_q05=int(
        (food_pairs[difference_q_columns] < 0.05).sum().sum()
    ),
    diff_min_q=float(food_pairs[difference_q_columns].min().min()),
    top_pair=[
        top_pairs["food_1"].iloc[0],
        top_pairs["food_2"].iloc[0],
        float(top_pairs["rho_All"].iloc[0]),
        float(top_pairs["q_All"].iloc[0]),
    ],
)

print(
    top_pairs[["food_1", "food_2", "rho_All", "q_All"]]
    .head(6)
    .round(3)
    .to_string(index=False)
)

# Create correlation heatmaps. Food order is determined by hierarchical clustering.
correlation_matrices = {
    group_name: group_data[foods].corr(method="spearman")
    for group_name, group_data in groups.items()
}

distance_matrix = 1 - correlation_matrices["All"].fillna(0).to_numpy()
np.fill_diagonal(distance_matrix, 0)

food_leaf_order = leaves_list(
    linkage(
        squareform((distance_matrix + distance_matrix.T) / 2, checks=False),
        method="average",
    )
)
ordered_foods = [foods[index] for index in food_leaf_order]

panel_names = ["All"] + diagnosis_order
n_rows = math.ceil(len(panel_names) / 2)

fig, axes = plt.subplots(n_rows, 2, figsize=(15, 6.75 * n_rows), squeeze=False)

for ax in axes.ravel()[len(panel_names):]:
    ax.axis("off")

for ax, group_name in zip(axes.ravel(), panel_names):
    matrix = correlation_matrices[group_name].loc[
        ordered_foods, ordered_foods
    ]

    image = ax.imshow(
        matrix.to_numpy(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(ordered_foods)))
    ax.set_yticks(range(len(ordered_foods)))
    ax.set_xticklabels(
        [display_food_name(food) for food in ordered_foods],
        rotation=90,
        fontsize=7.5,
    )
    ax.set_yticklabels(
        [display_food_name(food) for food in ordered_foods],
        fontsize=7.5,
    )

    significant_pairs = food_pairs.set_index(
        ["food_1", "food_2"]
    )[f"q_{group_name}"]

    for row, food_1 in enumerate(ordered_foods):
        for column, food_2 in enumerate(ordered_foods):
            if row == column:
                continue

            q_value = significant_pairs.get(
                (food_1, food_2),
                significant_pairs.get((food_2, food_1), 1),
            )

            if q_value < 0.05:
                ax.text(
                    column,
                    row,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    title = (
        "All participants"
        if group_name == "All"
        else group_name
    )
    ax.set_title(
        f"{title} (n = {len(groups[group_name])})",
        fontweight="bold",
    )

fig.colorbar(
    image,
    ax=axes.ravel().tolist(),
    shrink=0.5,
    label="Spearman rho",
)
fig.suptitle(
    "Food-food correlations of habitual intake "
    "(* BH q < 0.05 within group)",
    fontweight="bold",
)
plt.savefig(
    C.D1_FIG_DIR / "d1_food_correlations.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close()


# ---------------------------------------------------------------------
# 3. Identify dietary patterns with k-means clustering
# ---------------------------------------------------------------------

def cluster_participants(food_columns, number_of_clusters, seed=C.SEED):
    """Standardize participant food scores and fit a k-means model."""
    values = StandardScaler().fit_transform(
        participants[food_columns]
    )
    labels = KMeans(
        n_clusters=number_of_clusters,
        n_init=20,
        random_state=seed,
    ).fit_predict(values)

    return values, labels


# Compare silhouette scores for cluster counts 2 through 6.
silhouette_scores = {}

for number_of_clusters in C.K_RANGE:
    standardized_values, labels = cluster_participants(
        foods,
        number_of_clusters,
    )
    silhouette_scores[number_of_clusters] = float(
        silhouette_score(standardized_values, labels)
    )

best_cluster_count = max(
    silhouette_scores,
    key=silhouette_scores.get,
)

# Use K_CLUSTERS (config.py) for comparability, regardless of the best silhouette score.
number_of_clusters = C.K_CLUSTERS
standardized_values, cluster_labels = cluster_participants(
    foods,
    number_of_clusters,
)
participants["Cluster"] = cluster_labels + 1

# Project participant food scores into two PCA dimensions for plotting.
pca = PCA(n_components=2, random_state=C.SEED).fit(standardized_values)
pca_coordinates = pca.transform(standardized_values)

# Test whether cluster membership differs by diagnosis.
cluster_by_diagnosis = pd.crosstab(
    participants["diagnosis"],
    participants["Cluster"],
).loc[diagnosis_order]

chi_square, chi_square_p, _, _ = stats.chi2_contingency(
    cluster_by_diagnosis
)

# Monte Carlo permutation p-value for the diagnosis-cluster association.
permutation_rng = np.random.default_rng(C.SEED)
permutation_count = 0

for _ in range(C.N_PERM_CLUSTER):
    shuffled_diagnosis = permutation_rng.permutation(
        participants["diagnosis"].to_numpy()
    )
    permuted_table = pd.crosstab(
        shuffled_diagnosis,
        participants["Cluster"],
    )
    permuted_chi_square = stats.chi2_contingency(
        permuted_table
    )[0]

    permutation_count += permuted_chi_square >= chi_square - 1e-12

permutation_p = (permutation_count + 1) / (C.N_PERM_CLUSTER + 1)

cramers_v = float(
    np.sqrt(
        chi_square
        / (
            len(participants)
            * (min(cluster_by_diagnosis.shape) - 1)
        )
    )
)

# Estimate cluster stability with bootstrap samples.
bootstrap_ari = []

for bootstrap_index in range(C.N_BOOT):
    bootstrap_rng = np.random.default_rng(C.SEED + bootstrap_index)
    sample_indices = bootstrap_rng.choice(
        len(participants),
        len(participants),
        replace=True,
    )

    bootstrap_model = KMeans(
        n_clusters=number_of_clusters,
        n_init=5,
        random_state=bootstrap_index,
    ).fit(standardized_values[sample_indices])

    bootstrap_labels = bootstrap_model.predict(standardized_values)
    bootstrap_ari.append(
        adjusted_rand_score(cluster_labels, bootstrap_labels)
    )

# Sensitivity check using the smaller food set (D1_SENSITIVITY_FOODS).
values_small_set, labels_small_set = cluster_participants(
    C.D1_SENSITIVITY_FOODS,
    number_of_clusters,
)
ari_small_set_vs_full = float(
    adjusted_rand_score(cluster_labels, labels_small_set)
)

diagnosis_by_small_set_cluster = pd.crosstab(
    participants["diagnosis"],
    labels_small_set,
)
p_small_set = float(
    stats.chi2_contingency(diagnosis_by_small_set_cluster)[1]
)

# Summarize average food scores by cluster.
cluster_profiles = participants.groupby("Cluster")[foods].mean().T
cluster_profiles["overall"] = participants[foods].mean()

# Express each cluster's food scores as standard deviations from the overall mean.
profile_z_scores = (
    cluster_profiles.drop(columns="overall")
    .sub(cluster_profiles["overall"], axis=0)
    .div(participants[foods].std(), axis=0)
)

cluster_profiles.round(3).to_csv(
    C.D1_TAB_DIR / "d1_cluster_profiles.csv"
)
cluster_by_diagnosis.to_csv(
    C.D1_TAB_DIR / "d1_cluster_by_diagnosis.csv"
)
participants.to_csv(
    C.D1_TAB_DIR / "d1_participant_clusters.csv"
)

summary.update(
    silhouette=silhouette_scores,
    k_best_by_silhouette=best_cluster_count,
    k_used=number_of_clusters,
    cluster_sizes=participants["Cluster"]
    .value_counts()
    .sort_index()
    .to_dict(),
    cluster_vs_diagnosis_chi2=float(chi_square),
    cluster_vs_diagnosis_p_chi2=float(chi_square_p),
    cluster_vs_diagnosis_p_mc=float(permutation_p),
    cramers_v=cramers_v,
    bootstrap_ari_mean=float(np.mean(bootstrap_ari)),
    ari_small_set_vs_full=ari_small_set_vs_full,
    p_small_set_cluster_vs_diagnosis=p_small_set,
    cluster_top_foods={
        int(cluster): [
            f"{display_food_name(food)} "
            f"({profile_z_scores.loc[food, cluster]:+.2f} SD)"
            for food in profile_z_scores[cluster]
            .abs()
            .sort_values(ascending=False)
            .index[:3]
        ]
        for cluster in profile_z_scores.columns
    },
)

cluster_summary_keys = (
    "silh",
    "cluster",
    "cramer",
    "boot",
    "ari",
    "p_small",
)

print(
    json.dumps(
        {
            key: value
            for key, value in summary.items()
            if key.startswith(cluster_summary_keys)
        },
        indent=1,
        default=str,
    )
)


# ---------------------------------------------------------------------
# Plot cluster results
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 4.8),
    gridspec_kw={"width_ratios": [1.1, 1, 1.1]},
)

cluster_palette = ["#1F77B4", "#E67E22", "#2CA02C", "#9467BD",
                   "#D62728", "#8C564B", "#17BECF", "#7F7F7F"]
cluster_colors = {
    cluster: cluster_palette[cluster - 1]
    for cluster in range(1, number_of_clusters + 1)
}

# PCA scatter plot.
for cluster in sorted(participants["Cluster"].unique()):
    is_cluster = participants["Cluster"].to_numpy() == cluster
    cluster_size = int(is_cluster.sum())

    axes[0].scatter(
        pca_coordinates[is_cluster, 0],
        pca_coordinates[is_cluster, 1],
        color=cluster_colors[cluster],
        label=f"Cluster {cluster} (n={cluster_size})",
        s=34,
        alpha=0.85,
    )

axes[0].set_xlabel(
    f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}%)"
)
axes[0].set_ylabel(
    f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}%)"
)
axes[0].legend(fontsize=8)
axes[0].set_title(
    "Participants in diet PCA space",
    fontweight="bold",
)

# Diagnosis group composition by cluster.
(
    cluster_by_diagnosis
    .div(cluster_by_diagnosis.sum(axis=1), axis=0)
    .mul(100)
    .plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        color=[
            cluster_colors[cluster]
            for cluster in cluster_by_diagnosis.columns
        ],
        width=0.7,
    )
)

axes[1].set_ylabel("% of group")
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Cluster", fontsize=8)
axes[1].set_title(
    f"Clusters by diagnosis (V = {cramers_v:.2f}, "
    f"permutation p = {permutation_p:.3f})",
    fontweight="bold",
)

# Show the ten food items with the largest cluster deviations.
top_profile_foods = (
    profile_z_scores.abs()
    .max(axis=1)
    .sort_values(ascending=False)
    .index[:10]
)

sns.heatmap(
    profile_z_scores.loc[top_profile_foods].rename(
        index=display_food_name
    ),
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".1f",
    ax=axes[2],
    cbar=False,
    annot_kws={"fontsize": 8},
)
axes[2].set_title(
    "Cluster profile (SD from overall mean)",
    fontweight="bold",
)
axes[2].set_xlabel("Cluster")

plt.tight_layout()
plt.savefig(C.D1_FIG_DIR / "d1_clusters.png", dpi=170)
plt.close()


# Save the full summary.
with open(C.D1_DIR / "d1_summary.json", "w") as summary_file:
    json.dump(summary, summary_file, indent=2, default=str)