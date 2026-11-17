"""
We want to know how many motif-representative sequences it takes to reconstruct
the known motif. We generate some number of sequences from a motif's PWM, build a
motif from those sequences, and compare it to the known motif (Jensen-Shannon
divergence). As a negative baseline, we randomly draw the same number of sequences
from the TF's genomic non-motif set (nonmotif_negatives/<TF>_negatives.fa) and
compute the JSD of their PWM to the true motif.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import motifs
from scipy.spatial import distance

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import Motif
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "tables"
TAB_DIR.mkdir(exist_ok=True)


# %%
### Simulate sequences from motifs and sample real genomic negatives, computing
### JSD to the true motif PWM for a varying number of sequences
def read_fa(path):
    """Return the sequences in a FASTA file (one per record)."""
    return [line.strip() for line in open(path) if not line.startswith(">")]


motif_scores, non_motif_scores = {}, {}
num_seqs = [i + 5 for i in range(0, 100, 5)]
rng = np.random.default_rng(0)

# each TF's PWM lives in pwms/<Motif_ID>.txt; the Motif_ID is recorded per TF in
# cisbp_tfs.csv, and its genomic negatives in nonmotif_negatives/<TF>_negatives.fa
tf_df = pd.read_csv(f"{DATA_DIR}/sequences/cisbp/cisbp_tfs.csv")
pwm_dir = Path(f"{DATA_DIR}/sequences/cisbp/pwms")
neg_dir = Path(f"{DATA_DIR}/sequences/cisbp/nonmotif_negatives")

for row in tf_df.itertuples():
    neg_path = neg_dir / f"{row.TF_Name}_negatives.fa"
    if pd.isna(row.Motif_ID) or not neg_path.exists():
        continue  # no CisBP motif, or no negatives generated for this TF

    print(f"Simulating motif: {row.TF_Name}")
    motif = Motif(name=row.TF_Name, pwm_path=str(pwm_dir / f"{row.Motif_ID}.txt"))
    negatives = read_fa(neg_path)

    motif_scores[motif.name] = []
    non_motif_scores[motif.name] = []

    # Generate sequences and compute JSD for varying number of sequences
    for num in num_seqs:
        # motif-like: sequences generated from the PWM (best JSD of 100 samples)
        motif.generate_sequences(num, 100)
        motif_scores[motif.name].append(motif.seq_jsd)

        # non-motif: draw `num` real negatives, build a PWM, JSD to the true motif
        replace = num > len(negatives)
        sample = [
            negatives[i] for i in rng.choice(len(negatives), num, replace=replace)
        ]
        non_motif_pwm = motifs.create(sample).counts.normalize(motif.bg_freq)
        non_motif_scores[motif.name].append(
            distance.jensenshannon(  # Compute JSD to true motif
                motif.pwm.values.flatten(),
                np.array(list(non_motif_pwm.values())).flatten(),
            )
        )

motif_df = pd.DataFrame(motif_scores, index=num_seqs)
non_motif_df = pd.DataFrame(non_motif_scores, index=num_seqs)
motif_df.to_csv(f"{TAB_DIR}/motif_sim_scores.csv")
non_motif_df.to_csv(f"{TAB_DIR}/non_motif_sim_scores.csv")


# %%
### Look at pax2b in particular to see how the JSD changes as we generate more sequences
tf_df = pd.read_csv(f"{DATA_DIR}/sequences/cisbp/cisbp_tfs.csv")
pax2b_id = tf_df.set_index("TF_Name").loc["pax2b", "Motif_ID"]
motif = Motif(name="pax2b", pwm_path=f"{DATA_DIR}/sequences/cisbp/pwms/{pax2b_id}.txt")

# %%
jsds_10seqs = []
for i in range(50):
    motif.generate_sequences(10, 100)  # motif-like sequences
    print(f"Motif JSD: {motif.seq_jsd:.4f}")
    jsds_10seqs.append(motif.seq_jsd)
    for seq in motif.motif_seqs:
        print(seq)

# %%
jsds_30seqs = []
for i in range(50):
    motif.generate_sequences(30, 100)  # motif-like sequences
    print(f"Motif JSD: {motif.seq_jsd:.4f}")
    jsds_30seqs.append(motif.seq_jsd)

# %%
jsds_50seqs = []
for i in range(50):
    motif.generate_sequences(50, 100)  # motif-like sequences
    print(f"Motif JSD: {motif.seq_jsd:.4f}")
    jsds_50seqs.append(motif.seq_jsd)

# %%
# Plot box and whisker plot of JSDs for PWMs generated from 10 vs 30 vs 50 sequences
plt.figure(figsize=(5, 5))
data = [jsds_10seqs, jsds_30seqs, jsds_50seqs]
plt.boxplot(
    data,
    labels=["10", "30", "50"],
    patch_artist=True,
    boxprops=dict(facecolor="#738fc1", color="#738fc1", alpha=0.3),
    medianprops=dict(color="#738fc1", linewidth=2, alpha=0.8),
)
plt.xlabel("Number of sequences sampled")
plt.ylabel("JSD to Pax2b PWM")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig(f"{FIG_DIR}/pax2b_jsd_boxplot.svg", bbox_inches="tight")
plt.show()

# %%
### Look at means of JSDs for all TFs to see how many sequences it takes to reconstruct the motif
motif_df = pd.read_csv(f"{TAB_DIR}/motif_sim_scores.csv", index_col=0)
non_motif_df = pd.read_csv(f"{TAB_DIR}/non_motif_sim_scores.csv", index_col=0)

# Plot density curves of ipTM scores for motif vs non-motif sequences
fig, ax = plt.subplots(figsize=(5, 5))
n_seqs = motif_df.index.values  # [5, 10, 15, ..., 100]
motif_curves = motif_df.values.T  # Shape: (29 TFs, 20 timepoints)
nonmotif_curves = non_motif_df.values.T  # Shape: (29 TFs, 20 timepoints)

# Non-motif ribbon
nonmotif_min = nonmotif_curves.min(axis=0)
nonmotif_max = nonmotif_curves.max(axis=0)
nonmotif_mean = nonmotif_curves.mean(axis=0)
ax.fill_between(n_seqs, nonmotif_min, nonmotif_max, alpha=0.3, color="#d56c55")
ax.plot(
    n_seqs,
    nonmotif_mean,
    color="#d56c55",
    linewidth=2,
    alpha=0.8,
    label="Non-motif mean",
)

# Motif ribbon
motif_min = motif_curves.min(axis=0)
motif_max = motif_curves.max(axis=0)
motif_mean = motif_curves.mean(axis=0)
ax.fill_between(n_seqs, motif_min, motif_max, alpha=0.3, color="#738fc1")
ax.plot(n_seqs, motif_mean, color="#738fc1", linewidth=2, alpha=0.8, label="Motif mean")

# Verticle line at 30 sequences
ax.axvline(x=30, color="black", linestyle="--", linewidth=2, alpha=0.7)
ax.text(32, 0.22, "JSD stable ~30 sequences")

ax.set_xlabel("Number of sequences sampled")
ax.set_ylabel("JSD to true motif PWM")
ax.set_xlim(0, 105)
ax.set_ylim(0, 0.70)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.6, 0.95), loc="upper left")
plt.savefig(f"{FIG_DIR}/motif_sim_jsd.svg", bbox_inches="tight")
plt.show()

# %%
