"""
Mean motif ipTM x DBD pLDDT versus motif/non-motif AUROC, one point per TF.

Matches 01_conf-plots.py with SCORE_MODE = "rank": AUROC uses the geometric
mean of within-TF ipTM and DBD pLDDT percentile ranks across both classes.
The x-axis is the mean raw product ipTM * (dom_pLDDT / 100) over motif runs.
Pearson and Spearman correlations use two-sided p-values across TFs.
"""

# %%
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

RUN = "motif-nonmotif-genomic"
FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_STEM = FIG_DIR / "iptm_dbd_plddt_vs_auroc"

tf_df = pd.read_csv(DS["cisbp"].seq_csv)
tf_df["DBID"] = tf_df["DBID"].str.lower()
db_path = DATA_DIR / "jobs" / RUN / "jobdb.sqlite"
with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as conn:
    runs_df = pd.read_sql_query(
        "SELECT protein, motif, ipTM, dom_pLDDT FROM runs", conn
    )
runs_df = runs_df.merge(
    tf_df[["DBID", "TF_Name"]],
    left_on="protein",
    right_on="DBID",
    how="left",
    validate="many_to_one",
)
if runs_df["TF_Name"].isna().any():
    raise ValueError("Some run proteins have no matching TF name in the CisBP table")
if not np.isfinite(runs_df[["ipTM", "dom_pLDDT"]].to_numpy()).all():
    raise ValueError("ipTM and DBD pLDDT must be finite for every prediction")
runs_df["is_motif"] = ~runs_df["motif"].str.contains("_neg", regex=False)
runs_df["raw_product"] = runs_df["ipTM"] * (runs_df["dom_pLDDT"] / 100)

# %%
rows = []
for tf_name, group in runs_df.groupby("TF_Name", sort=True):
    is_motif = group["is_motif"]
    if is_motif.nunique() != 2:
        continue
    rank_score = np.sqrt(
        group["ipTM"].rank(pct=True) * group["dom_pLDDT"].rank(pct=True)
    )
    rows.append(
        {
            "TF_name": tf_name,
            "protein": group["protein"].iloc[0],
            "mean_motif_iptm_x_dbd_plddt": group.loc[is_motif, "raw_product"].mean(),
            "AUROC": roc_auc_score(is_motif, rank_score),
            "n_motif": int(is_motif.sum()),
            "n_nonmotif": int((~is_motif).sum()),
        }
    )
plot_df = pd.DataFrame(rows).sort_values("mean_motif_iptm_x_dbd_plddt")
x = plot_df["mean_motif_iptm_x_dbd_plddt"]
y = plot_df["AUROC"]
pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)
fit = stats.linregress(x, y)

plot_df.to_csv(f"{OUTPUT_STEM}_data.csv", index=False)
correlations = pd.DataFrame(
    [
        {"method": "Pearson", "correlation": pearson_r, "p_value": pearson_p},
        {"method": "Spearman", "correlation": spearman_r, "p_value": spearman_p},
    ]
)
correlations["n_tfs"] = len(plot_df)
correlations["auroc_score"] = "sqrt(ipTM percentile * DBD pLDDT percentile)"
correlations.to_csv(f"{OUTPUT_STEM}_correlations.csv", index=False)
print(correlations.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(
    x,
    y,
    s=75,
    color="#738fc1",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85,
    zorder=3,
)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(
    x_line,
    fit.slope * x_line + fit.intercept,
    color="#ef4444",
    linestyle="--",
    linewidth=1.8,
)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.text(
    0.04,
    0.96,
    f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.2e}\n"
    f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.2e}\n"
    f"n = {len(plot_df)} TFs",
    transform=ax.transAxes,
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.95),
)
label_tfs = {"sp2", "pax2a", "pax2b", "prkrira"}
for row in plot_df.itertuples(index=False):
    if row.TF_name in label_tfs:
        ax.annotate(
            row.TF_name.capitalize(),
            xy=(row.mean_motif_iptm_x_dbd_plddt, row.AUROC),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
ax.set_xlabel("Mean motif ipTM × (DBD pLDDT / 100)")
ax.set_ylabel("AUROC (motif vs non-motif)")
ax.set_title("Confidence vs. motif/non-motif discrimination", pad=12)
ax.set_ylim(0.2, 1.04)
ax.spines[["top", "right"]].set_visible(False)
fig.text(
    0.5,
    0.02,
    "AUROC score: √(ipTM percentile × DBD pLDDT percentile), ranked within each TF.",
    ha="center",
    fontsize=8,
    color="0.35",
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
fig.savefig(f"{OUTPUT_STEM}.svg", bbox_inches="tight")
fig.savefig(f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")
plt.show()
