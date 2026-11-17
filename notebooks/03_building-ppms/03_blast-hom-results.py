"""
Plot the average DNA-binding-domain similarity of zebrafish TFs whose ZFIN-curated human
ortholog has a JASPAR motif, per TF family. Only TFs with measured similarity contribute to
the averages. Hom_Sim is written to the reps csv by data/tools/blast/hom_results.py
(single-DBD families, BLAST) and by zfc2h2_hom.py / hd_hom.py / zbtb_hom.py (recognition-code
families, starred).
"""

# %%
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "figures"
FIG_DIR.mkdir(exist_ok=True)

dataset = DS["tfdb4_rep"]

# Families whose homolog/motif assignments come from recognition-code scripts
RECOG_CODE_FAMILIES = {"zf-C2H2", "Homeobox", "ZBTB"}


def get_no_match_families(tf_df: pd.DataFrame, results: dict) -> list[str]:
    """
    Return selected families that have no measured similarity in results.

    Args:
        tf_df: DataFrame with TF info, must include 'Family'.
        results: Dictionary of per-family similarities.

    Returns:
        List of family names with no measured similarity.
    """

    all_families = {str(f) for f in tf_df["Family"].dropna().unique()}
    return sorted(all_families - set(results.keys()) - {"nan"})


def plot_sim_results(results: dict, total_tfs: int):
    """
    Plot average similarity to JASPAR-bearing ZFIN orthologs per TF family.
    Families with no measured similarity contribute no bar.

    Args:
        results: {family: [sim, ...]} for selected TFs with measured similarity.
        total_tfs: Number of selected TFs, including those with missing similarity.
    """
    avg_families, std_families, n_families = {}, {}, {}
    for fam, sims in results.items():
        if fam == "nan":
            continue
        avg_sim = sum(sims) / len(sims)
        avg_families[fam] = avg_sim
        std_families[fam] = (sum((s - avg_sim) ** 2 for s in sims) / len(sims)) ** 0.5
        n_families[fam] = len(sims)

    sorted_avg = dict(sorted(avg_families.items(), key=lambda x: x[1], reverse=True))
    labels = list(sorted_avg.keys())
    sims = [p * 100 for p in sorted_avg.values()]
    errors = [std_families[f] * 100 for f in labels]

    norm = plt.Normalize(min(sims), max(sims))
    colors = cm.viridis(norm(sims))

    plt.figure(figsize=(14, 8))
    plt.bar(
        labels,
        sims,
        yerr=errors,
        color=colors,
        edgecolor="white",
        capsize=4,
        alpha=0.9,
        error_kw={"ecolor": "grey", "elinewidth": 1.5},
    )

    # n is the number of family members averaged into the bar.
    tick_labels = [
        f"{f}* (n={n_families[f]})"
        if f in RECOG_CODE_FAMILIES
        else f"{f} (n={n_families[f]})"
        for f in labels
    ]

    plt.xlabel("TF Family")
    plt.ylabel("Average Similarity / Identity (%)")
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=90, ha="right")
    plt.tick_params(axis="x", pad=15)  # drop the labels clear of the bars
    # Default x-margins scale with the number of families and leave a wide gap between
    # the y tick labels and the first bar; pad by a quarter bar instead.
    plt.xlim(-0.6, len(labels) - 0.4)
    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["bottom"].set_color("#CCCCCC")
    plt.figtext(
        0.01,
        0.01,
        f"Measured similarity: {sum(n_families.values())} / {total_tfs} TFs with a JASPAR ortholog",
        ha="left",
        fontsize=8,
        color="grey",
    )
    plt.figtext(
        0.99,
        0.01,
        "* recognition code identity",
        ha="right",
        fontsize=8,
        color="grey",
    )
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.savefig(f"{FIG_DIR}/blast_sim.svg", dpi=300)
    plt.show()


# %%
# Plot from saved dataframe (skips re-running the analysis)
tf_df = pd.read_csv(dataset.seq_csv, sep=",", header=0)
orthos_df = pd.read_csv(DATA_DIR / "tools/blast/zfin/ensdarg_human_orthos.csv")
jaspar_ortholog_ids = set(
    orthos_df.loc[orthos_df["jaspar_ids"].fillna("").str.strip().ne(""), "ensdarg"]
)
tf_df = tf_df.loc[tf_df["DBID"].isin(jaspar_ortholog_ids)].copy()
scored_tf_df = tf_df.dropna(subset=["Hom_Sim"])
print(f"TFs with a JASPAR-bearing ZFIN human ortholog: {len(tf_df)}")
print(f"TFs with measured similarity (plotted): {len(scored_tf_df)}")
print(f"TFs missing similarity (not plotted): {len(tf_df) - len(scored_tf_df)}")
results_from_df = {}
for _, row in scored_tf_df.iterrows():
    results_from_df.setdefault(str(row["Family"]), []).append(row["Hom_Sim"] / 100)
no_match = get_no_match_families(tf_df, results_from_df)
print(f"Selected families with no measured similarity (not plotted): {no_match}")
plot_sim_results(results_from_df, total_tfs=len(tf_df))

# %%
