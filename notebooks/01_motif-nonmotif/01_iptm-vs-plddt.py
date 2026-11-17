"""
Here we want to get the pLDDT of the DNA-binding domains as labeled by InterProScan domains with
relatively high PAE values to the DNA from the AF3 predictions. For each prediction, we will
compare the ipTM score to the average pLDDT of the DNA-binding domains.
"""

# %%
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.tfactor import load_tfs
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Load TF information
dataset = DS["cisbp"]
tf_df = pd.read_csv(dataset.seq_csv)
tfs = load_tfs(tf_df, name="DBID")

# Read job database
run = "motif-nonmotif"
db = Database(
    job_path=f"{DATA_DIR}/jobs/{run}",
    dataset=dataset.name,
)

# Query database to get ipTM scores and collect by motif
query = """
SELECT motif, ipTM, dom_pLDDT, dna_pLDDT
FROM runs
"""
scores = defaultdict(list)
for motif, iptm, dom_plddt, dna_plddt in db.cursor.execute(query).fetchall():
    scores[motif].append((iptm, dom_plddt, dna_plddt))

db.conn.close()

# %%
for tf_name in tf_df["TF_Name"]:
    # Get scores
    motif_scores = scores[tf_name]
    nonmotif_scores = scores[tf_name + "-non"]
    if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
        continue

    # Prepare data for plotting
    x1, y1 = zip(*[(iptm, dom_plddt) for iptm, dom_plddt, dna_plddt in motif_scores])
    x2, y2 = zip(*[(iptm, dom_plddt) for iptm, dom_plddt, dna_plddt in nonmotif_scores])
    corr = stats.spearmanr(a=x1 + x2, b=y1 + y2, alternative="greater")

    # Plot ipTM vs. dom_pLDDT colored by motif/non-motif
    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1, label="motif", alpha=0.7)
    plt.scatter(x2, y2, label="non-motif", alpha=0.7)
    plt.xlabel("ipTM")
    plt.ylabel("DBD-Domain plDDT")
    plt.title(f"{tf_name.capitalize()}: ipTM vs. DBD-domain plDDT")
    plt.text(
        0.05,
        0.95,
        f"Spearman r = {corr.correlation:.2f}\np-value = {corr.pvalue:.2e}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

    # Prepare data for plotting
    x1, y1 = zip(*[(iptm, dna_plddt) for iptm, dom_plddt, dna_plddt in motif_scores])
    x2, y2 = zip(*[(iptm, dna_plddt) for iptm, dom_plddt, dna_plddt in nonmotif_scores])
    corr = stats.spearmanr(a=x1 + x2, b=y1 + y2, alternative="greater")

    # Plot ipTM vs. dna_pLDDT colored by motif/non-motif
    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1, label="motif", alpha=0.7)
    plt.scatter(x2, y2, label="non-motif", alpha=0.7)
    plt.xlabel("ipTM")
    plt.ylabel("DNA plDDT")
    plt.title(f"{tf_name.capitalize()}: ipTM vs. DNA plDDT")
    plt.text(
        0.05,
        0.95,
        f"Spearman r = {corr.correlation:.2f}\np-value = {corr.pvalue:.2e}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend(loc="lower right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

# %%
# Compute per-TF AUROC (motif vs. non-motif discrimination) for ipTM, DBD pLDDT,
# and DNA pLDDT. AUROC = P(motif metric > non-motif metric); 0.5 = no discrimination.
label_tfs = {"sp2", "pax2a", "pax2b", "prkrira", "mef2aa"}


def auroc(pos, neg):
    """AUROC for separating positives (motif) from negatives (non-motif)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return u / (len(pos) * len(neg))


tf_names, auroc_iptm, auroc_dbd, auroc_dna = [], [], [], []

for tf_name in tf_df["TF_Name"]:
    motif_scores = scores[tf_name]
    nonmotif_scores = scores[tf_name + "-non"]
    if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
        continue

    tf_names.append(tf_name)
    auroc_iptm.append(
        auroc([s[0] for s in motif_scores], [s[0] for s in nonmotif_scores])
    )
    auroc_dbd.append(
        auroc([s[1] for s in motif_scores], [s[1] for s in nonmotif_scores])
    )
    auroc_dna.append(
        auroc([s[2] for s in motif_scores], [s[2] for s in nonmotif_scores])
    )

for ylabel, auroc_plddt, title in [
    (
        "AUROC (DBD plDDT)",
        auroc_dbd,
        "ipTM vs. DBD plDDT discrimination",
    ),
    (
        "AUROC (DNA plDDT)",
        auroc_dna,
        "ipTM vs. DNA plDDT discrimination",
    ),
]:
    corr = stats.spearmanr(a=auroc_iptm, b=auroc_plddt, alternative="greater")

    plt.figure(figsize=(9, 6))
    plt.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")
    plt.scatter(
        auroc_iptm, auroc_plddt, alpha=0.8, color="#738fc1", edgecolor="black", s=80
    )
    for name, dx, dy in zip(tf_names, auroc_iptm, auroc_plddt):
        if name in label_tfs:
            plt.annotate(
                name.capitalize(),
                (dx, dy),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=8,
            )
    plt.xlabel("AUROC (ipTM)")
    plt.ylabel(ylabel)
    plt.text(
        0.05,
        0.95,
        f"Spearman r = {corr.correlation:.2f}\np-value = {corr.pvalue:.2e}",
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(
        f"{FIG_DIR}/ipTM_vs_{ylabel.replace(' ', '_')}.svg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%
