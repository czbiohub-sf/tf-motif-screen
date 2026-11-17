"""
Plot motif similarity distributions for HOMER motifs, comparing de novo vs. JASPAR matches.

Ben Iovino 5/1/26 Biohub
"""

# %%
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% Load and parse motifs
df = pd.read_csv(
    f"{DATA_DIR}/tools/homer/motifs.tsv",
    sep="\t",
    header=0,
    names=["consensus", "motif", "source_cluster", "source_clusters"],
)


# Parse BestGuess entries from motif column
def parse_best_guesses(motif_str):
    hits = re.findall(r"BestGuess:(.+?)\(([0-9]+\.[0-9]+)\)", motif_str)
    results = []
    for name, score in hits:
        results.append({"name": name, "score": float(score)})
    return results


parsed = df["motif"].apply(parse_best_guesses)

# Take the first BestGuess per motif
records = []
for i, (consensus, guesses) in enumerate(zip(df["consensus"], parsed)):
    if not guesses:
        continue
    first = guesses[0]
    # Extract short TF name: first path component before the score
    tf_name = first["name"].split("/")[0]
    records.append(
        {
            "consensus": consensus,
            "tf_name": tf_name,
            "best_score": first["score"],
            "source_cluster": df.iloc[i]["source_cluster"],
        }
    )
plot_df = pd.DataFrame(records).sort_values("best_score", ascending=False)

# JASPAR motifs start with "MA"; de novo motifs start with a digit
plot_df["is_jaspar"] = plot_df["consensus"].str.match(r"^MA")

# %% Score distribution density plots
fig, ax = plt.subplots(figsize=(5, 5))
counts = plot_df["is_jaspar"].value_counts()
plot_df["type"] = plot_df["is_jaspar"].map(
    {
        True: f"JASPAR (n={counts.get(True, 0)})",
        False: f"De Novo (n={counts.get(False, 0)})",
    }
)
palette = {
    f"JASPAR (n={counts.get(True, 0)})": "#87a9d4",
    f"De Novo (n={counts.get(False, 0)})": "#f4a460",
}
sns.kdeplot(
    data=plot_df,
    x="best_score",
    hue="type",
    ax=ax,
    palette=palette,
    fill=True,
    alpha=0.4,
    common_norm=True,
)
ax.set_xlabel("HOMER similarity")
ax.set_ylabel("Density")
ax.set_title("PWM Similarity to Closest JASPAR Match")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/score_dist.svg", dpi=300)
plt.show()

# %%
