"""
Here we are looking at the paired aligned error (PAE) of AlphaFold3 predictions as heatmaps. The
PAE can help us find regions of the structure that may be in contact with each other, via regions
of low PAE values.

Ben Iovino 3/4/2026 Biohub
"""

# %%
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser

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
dom_df = pd.read_csv(dataset.dom_csv)
tfs = load_tfs(tf_df, name="DBID")

# Read job database
run = "motif-nonmotif"
db = Database(
    job_path=f"{DATA_DIR}/jobs/{run}",
    dataset=dataset.name,
)

# Query database to get structure paths for motifs of interest
query = """
SELECT structure_path, motif, iptm, dom_plddt
FROM runs
"""
structures = defaultdict(list)
for structure_path, motif, iptm, dom_plddt in db.cursor.execute(query).fetchall():
    structures[motif].append((structure_path, iptm, dom_plddt))

db.conn.close()

# %%
# Plot 2D heatmap
tf = "pax2a"
for structure_path, iptm, dom_plddt in structures[f"{tf}"]:
    # print(iptm, dom_plddt, structure_path)
    structure_path = Path(structure_path)
    run = structure_path.parent.parent.name
    if not run.endswith("_136"):
        continue

    with open(
        structure_path.parent / (structure_path.parent.name + "_confidences.json")
    ) as f:
        pae = np.array(json.load(f)["pae"])
    fig, ax = plt.subplots()
    im = ax.imshow(pae, cmap="Greens_r")
    cbar = fig.colorbar(im, ax=ax, location="top", pad=0.02, shrink=0.74)
    cbar.set_label("Expected positional error (Å)")
    ax.set_xlabel("Scored residue (measured)")
    ax.set_ylabel("Aligned residue (anchor)")
    # fig.savefig(f"{FIG_DIR}/pax2a_pae.svg", dpi=300)
    plt.show()

# %%
# Get DNA positions
parser = MMCIFParser()
structure = parser.get_structure("model", structure_path)
mmcif_dict = parser._mmcif_dict
dna_indices, index = [], 0
dna_res = ["DA", "DT", "DC", "DG"]
for model in structure:
    for chain in model:
        for residue in chain:
            index += 1
            if residue.resname.strip() in dna_res:
                dna_indices.append(index)

# Plot contact counts per residue across all DNA nucleotides
dna_idx = np.array(dna_indices) - 1
b_pae = pae[dna_idx, : dna_idx[0]]  # shape (n_dna, n_protein)
contact_counts = (b_pae < 5).sum(
    axis=0
)  # number of DNA nucleotides with PAE < 5 per residue
nonzero = contact_counts[contact_counts > 0]
mean_threshold = nonzero.mean() if len(nonzero) > 0 else 0

plt.plot(contact_counts, color="#738fc1", alpha=0.8)
plt.xlabel("Residue index")
plt.ylabel("Confidently positioned nucleotides (PAE < 5Å)")
plt.axhline(
    mean_threshold,
    color="#d56c55",
    linestyle="--",
    label=f"Mean contact frequency ({mean_threshold:.1f})",
)
plt.legend()
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.savefig(f"{FIG_DIR}/pax2a_dna_counts.svg", dpi=300)
plt.show()

# %%
