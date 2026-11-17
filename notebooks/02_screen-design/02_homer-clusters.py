"""
Comparing all de novo HOMER motifs and significant JASPAR hits per cluster.

Ben Iovino 4/23/26  Biohub
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %%
output_dir = Path(f"{DATA_DIR}/tools/homer/jaspar/output")
p_threshold = 1e-12
plot_cluster = 12  # set to None to plot all clusters

clusters = sorted(
    [d for d in output_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name)
)

homer_counts = []
known_counts = []
cluster_ids = []

for cluster_dir in clusters:
    cid = int(cluster_dir.name)

    # Count all de novo motifs, excluding reverse-complement and similar variants.
    homer_total = 0
    homer_results_dir = cluster_dir / "homerResults"
    if homer_results_dir.exists():
        for motif_file in homer_results_dir.glob("motif*.motif"):
            if "RV" in motif_file.name or "similar" in motif_file.name:
                continue
            homer_total += 1

    # Count significant known results
    known_sig = 0
    known_file = cluster_dir / "knownResults.txt"
    if known_file.exists():
        with open(known_file) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    pval = float(parts[2])
                    if pval < p_threshold:
                        known_sig += 1
                except ValueError:
                    continue

    cluster_ids.append(cid)
    homer_counts.append(homer_total)
    known_counts.append(known_sig)

# %%
if plot_cluster is not None:
    idx = cluster_ids.index(plot_cluster)
    cluster_ids = [cluster_ids[idx]]
    homer_counts = [homer_counts[idx]]
    known_counts = [known_counts[idx]]

x = range(len(cluster_ids))

figwidth = 2.2 if plot_cluster is not None else 8
fig, ax = plt.subplots(figsize=(figwidth, 4))
ax.bar(
    list(x),
    known_counts,
    width=0.7,
    label="JASPAR\n(p < 10⁻¹²)" if plot_cluster is not None else "JASPAR (p < 10⁻¹²)",
    alpha=0.6,
    color="#87a9d4",
)
ax.bar(
    list(x),
    homer_counts,
    bottom=known_counts,
    width=0.7,
    label="De novo (all)",
    alpha=0.6,
    color="#f4a460",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks(list(x))
ax.set_xticklabels(cluster_ids, fontsize=8)
ax.set_xlim(-0.5, len(cluster_ids) - 0.5)
ax.set_xlabel("Cluster", fontsize=9)
ax.set_ylabel("Motifs per cluster", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.set_title(
    "HOMER motif hits\nper cluster"
    if plot_cluster is not None
    else "HOMER motif hits per cluster"
)
ax.legend(reverse=True, fontsize=9 if plot_cluster is not None else 10)
if plot_cluster is not None:
    ax.set_ylim(0, 1.35 * (known_counts[0] + homer_counts[0]))
plt.tight_layout()
out_name = (
    f"cluster_{plot_cluster}_hits" if plot_cluster is not None else "cluster_hits"
)
plt.savefig(f"{FIG_DIR}/{out_name}.svg")
plt.savefig(f"{FIG_DIR}/{out_name}.png", dpi=300)
plt.show()

# %%
