"""
How motif-like are the generated non-motif (negative) sequences, really?

For each TF we build a PWM from its 150 generated non-motif sequences
(nonmotif_negatives/<TF>_negatives.fa) and compare it to the true PWM in the
.motif file using Jensen-Shannon divergence -- exactly the metric in
notebooks/01_motif-nonmotif/01_motif-sim.py (HOMER-normalized PWM, bg_freq={A:.3,C:.2,G:.2,T:.3},
JSD over the flattened width x 4 matrix). HIGH JSD = dissimilar = a good negative.

References for scale:
  - positive control: PWM from sequences sampled FROM the true motif (low JSD).
  - the paired <TF>-non.motif PWM (the previous negative construction), where present.
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

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "tables"
TAB_DIR.mkdir(exist_ok=True)

BG = {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
NEG_DIR = Path(f"{DATA_DIR}/sequences/cisbp/nonmotif_negatives")
MOT_DIR = Path(f"{DATA_DIR}/sequences/cisbp/motifs")
# degenerate TFs whose high-IC core was < 4 bp and so went unfiltered
UNFILTERABLE = {
    "arid6",
    "dharma",
    "kdm2aa",
    "kdm2ba",
    "lin54",
    "pax2b",
    "phf21ab",
    "pou3f3b",
    "prkrirb",
    "si:dkey-43p13.5",
}
rng = np.random.default_rng(0)


def pwm_from_seqs(seqs):
    """HOMER-normalized position-probability matrix from equal-length seqs."""
    pfm = motifs.create(seqs).counts.normalize(BG)
    return pd.DataFrame(pfm, columns=["A", "C", "G", "T"])


def jsd(true_pwm, other_pwm):
    return distance.jensenshannon(true_pwm.values.flatten(), other_pwm.values.flatten())


def read_fa(path):
    return [line.strip() for line in open(path) if not line.startswith(">")]


# %%
### Per-TF JSD: non-motif PWM vs true PWM, with positive control + old -non PWM
rows = []
for fa in sorted(NEG_DIR.glob("*_negatives.fa")):
    name = fa.name[: -len("_negatives.fa")]
    motif = Motif(name=name, pwm_path=MOT_DIR / f"{name}.motif")
    true_pwm = motif.pwm  # k x 4, per-position probs
    k = true_pwm.shape[0]

    # (1) our generated non-motif sequences
    neg_seqs = read_fa(fa)
    jsd_neg = jsd(true_pwm, pwm_from_seqs(neg_seqs))

    # (2) positive control: sequences sampled from the true motif (single 150-draw,
    #     averaged over 20 replicates for a stable reference)
    pos_jsds = []
    for _ in range(20):
        sampled = [
            "".join(
                rng.choice(["A", "C", "G", "T"], p=true_pwm.iloc[j].values)
                for j in range(k)
            )
            for _ in range(150)
        ]
        pos_jsds.append(jsd(true_pwm, pwm_from_seqs(sampled)))
    jsd_pos = float(np.mean(pos_jsds))

    # (3) the previous -non.motif construction, for reference (JSD of its PWM directly)
    non_path = MOT_DIR / f"{name}-non.motif"
    if non_path.exists():
        non_pwm = Motif(name=name + "-non", pwm_path=non_path).pwm
        jsd_oldnon = jsd(true_pwm, non_pwm)
    else:
        jsd_oldnon = np.nan

    rows.append(
        {
            "tf": name,
            "k": k,
            "filtered": name not in UNFILTERABLE,
            "jsd_nonmotif": jsd_neg,
            "jsd_positive_ctrl": jsd_pos,
            "jsd_old_non": jsd_oldnon,
        }
    )
    print(
        f"{name:16s} k={k:2d} filt={name not in UNFILTERABLE!s:5s} "
        f"JSD(neg)={jsd_neg:.3f}  JSD(pos)={jsd_pos:.3f}  JSD(old-non)={jsd_oldnon:.3f}"
    )

df = pd.DataFrame(rows).sort_values("jsd_nonmotif", ascending=False)
df.to_csv(f"{TAB_DIR}/nonmotif_sim_jsd.csv", index=False)

print("\n=== summary (JSD to true PWM; higher = more dissimilar = better negative) ===")
for col in ["jsd_nonmotif", "jsd_positive_ctrl", "jsd_old_non"]:
    print(f"  {col:20s} mean={df[col].mean():.3f}  median={df[col].median():.3f}")
print(
    "\n  filtered TFs   : JSD(neg) mean = "
    f"{df.loc[df.filtered,'jsd_nonmotif'].mean():.3f}"
)
print(
    "  unfilterable   : JSD(neg) mean = "
    f"{df.loc[~df.filtered,'jsd_nonmotif'].mean():.3f}"
)

# %%
### Plot: per-TF non-motif JSD vs positive control
set_plotting_style()

d = df.sort_values("jsd_nonmotif")
y = np.arange(len(d))
fig, ax = plt.subplots(figsize=(6, 8))
ax.hlines(y, d["jsd_positive_ctrl"], d["jsd_nonmotif"], color="lightgray", zorder=1)
ax.scatter(
    d["jsd_positive_ctrl"],
    y,
    color="#738fc1",
    label="positive control (sampled from motif)",
    zorder=2,
)
ax.scatter(
    d["jsd_nonmotif"], y, color="#d56c55", label="non-motif (our negatives)", zorder=2
)
# mark unfilterable TFs
for yi, filt in zip(y, d["filtered"]):
    if not filt:
        ax.text(-0.01, yi, "*", va="center", ha="right", color="black")
ax.set_yticks(y)
ax.set_yticklabels(d["tf"], fontsize=7)
ax.set_xlabel("JSD to true motif PWM")
ax.set_title(
    "Non-motif sequences are far from the true motif\n(* = degenerate core, unfiltered)"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/nonmotif_sim_jsd.svg")
plt.show()
