"""
Per-TF prediction confidence broken down by TF family.

Ben Iovino 7/16/26 Biohub
"""

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%
# Family lookup, restricted to the reps set
dataset = DS["tfdb4_rep"]
rep_tfs = pd.read_csv(dataset.seq_csv)
rep_tfs["protein"] = rep_tfs["DBID"].str.lower()
family_lookup = dict(zip(rep_tfs["protein"], rep_tfs["Family"]))
rep_proteins = set(rep_tfs["protein"])

# Pull every prediction with its confidence metrics
db = Database(job_path=f"{DATA_DIR}/jobs/hero", dataset=dataset.name)
runs_df = pd.read_sql("SELECT protein, ipTM, dom_pLDDT FROM runs", db.conn)
db.conn.close()

# Ignore TFs that are not in the tfdb4_rep set
n_before = runs_df["protein"].nunique()
runs_df = runs_df[runs_df["protein"].isin(rep_proteins)].copy()
print(
    f"Restricted to tfdb4_rep: {runs_df['protein'].nunique()}/{n_before} proteins kept"
)

# Drop predictions where no DNA-binding domain could be placed (dom_pLDDT == 0).
n_tf_before = runs_df["protein"].nunique()
runs_df = runs_df[runs_df["dom_pLDDT"] > 0].copy()
print(
    f"Dropped dom_pLDDT==0 (no DBD placed): "
    f"{runs_df['protein'].nunique()}/{n_tf_before} proteins retain >=1 prediction"
)

# Combined confidence: ipTM (0-1) times domain pLDDT rescaled to 0-1.
runs_df["combined"] = runs_df["ipTM"] * (runs_df["dom_pLDDT"] / 100)
per_tf = (  # Per-TF mean combined confidence over all its predictions.
    runs_df.groupby("protein")
    .agg(combined_total=("combined", "mean"), n=("combined", "size"))
    .reset_index()
)
per_tf["family"] = per_tf["protein"].map(family_lookup)

# %%
# Boxplot broken down by family: one box per family of the per-TF combined confidence.
# Pick the 20 biggest families by TF count from the full tfdb4 set (not the reps).
all_tfs = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
fam_counts = all_tfs["Family"].value_counts().drop("Others", errors="ignore")
FAM_ORDER = fam_counts.head(16).index.tolist()
fam_sizes = per_tf["family"].value_counts().reindex(FAM_ORDER).fillna(0).astype(int)

by_fam = per_tf[per_tf["family"].isin(FAM_ORDER)]
print(
    f"Boxplot by family: showing {len(FAM_ORDER)} families "
    f"({int(fam_sizes.sum())} TFs)."
)

# Box plots
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=by_fam,
    x="family",
    y="combined_total",
    order=FAM_ORDER,
    width=0.6,
    linewidth=0.9,
    fliersize=2,
    color="#4C72B0",
    ax=ax,
)

ax.set_xlabel("TF Family")
ax.set_ylabel("Mean per-TF ipTM x DBD pLDDT")
ax.set_ylim(0, 1)
ax.set_xticks(range(len(FAM_ORDER)))
ax.set_xticklabels(FAM_ORDER, rotation=45, ha="right")
ax.set_title("Per-family confidence ")
ax.grid(True, axis="y", linestyle="-", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fam_conf.svg", dpi=300)

# %%
