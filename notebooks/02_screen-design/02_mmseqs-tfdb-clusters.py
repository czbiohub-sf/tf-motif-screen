"""
Plots for the per-family clusters returned by mmseqs2 on the TFDB dataset: representatives vs.
all sequences per family, and intra- vs inter-cluster sequence similarity. If the similarity
between sequences in a cluster is relatively high, while the similarity between clusters is
relatively lower, then this suggests that the clusters may represent different domain
architectures within the same family.

The similarities are computed with MAFFT by data/tools/mmseqs/tfdb_clusters.py, which writes
notebooks/reports/intra_inter_similarities.csv.
"""

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%
# Count representatives vs. all sequences for each family
tfs = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
rep_tfs = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
rep_tfs = rep_tfs[~rep_tfs["DBID"].str.endswith(("_1", "_2", "_3"), na=False)]

# Rank by total TF count, keeping first-seen family order for ties.
all_counts = (
    tfs["Family"].value_counts(sort=False).sort_values(ascending=False, kind="stable")
)
rep_counts = (
    rep_tfs["Family"].value_counts().reindex(all_counts.index).fillna(0).astype(int)
)
df = pd.DataFrame({"all": all_counts, "rep": rep_counts})
# Show the top30 families by total TF count, excluding uncategorized entries.
df = df.drop(index=["Other", "Others"], errors="ignore").head(30)
family_order = df.index.tolist()

# Draw "all" bar first (wider, muted), then "rep" bar on top
fig, ax = plt.subplots(figsize=(7, 18))
y = np.arange(len(df))
ax.barh(y, df["all"], color="#92c5de", label=f"All TFs (n={len(tfs):,})", zorder=2)
ax.barh(y, df["rep"], color="#2166ac", label=f"Reps (n={len(rep_tfs):,})", zorder=3)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(family_order)
ax.set_xlabel("Sequence count (log scale)")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(ticker.NullFormatter())

# Annotate all families with rep/all counts, just right of each bar
for i, (fam, row) in enumerate(df.iterrows()):
    ax.text(
        row["all"] * 1.05,
        i,
        f"{int(row['rep'])}/{int(row['all'])}",
        va="center",
        ha="left",
        color="#333333",
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.xaxis.grid(True, which="both", color="#e8e8e8", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(frameon=False)
ax.invert_yaxis()
ax.margins(y=0)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/family_counts.svg", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/family_counts.png", dpi=300, bbox_inches="tight")

# %%
tfs = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
sim_df = pd.read_csv(f"{PROJECT_ROOT}/notebooks/reports/intra_inter_similarities.csv")
sim_df["family"] = sim_df["family"].str.replace(".recog_all_seqs", "", regex=False)

# Reuse the exact selection and order shown in the family-count plot.

fig, ax = plt.subplots(figsize=(6, 18))
sns.violinplot(
    data=sim_df,
    y="family",
    x="similarity",
    hue="type",
    order=family_order,
    split=False,
    density_norm="width",
    inner="quart",
    inner_kws={"linewidth": 1, "color": "black"},
    ax=ax,
)
ax.set_ylabel("Family")
ax.set_xlabel("Percent Similarity (%)")
ax.set_title("Intra- vs Inter-cluster Sequence Similarity by TF Family")
ax.set_xlim(0, 100)
plt.yticks(fontsize=10)
plt.legend(title="Similarity type")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/sim_boxplot.svg", dpi=300)
plt.savefig(f"{FIG_DIR}/sim_boxplot.png", dpi=300)
plt.show()
