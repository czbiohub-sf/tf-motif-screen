"""
Compare the two folding passes by how much the second pass contributes to each
TF's top-ranked structures, and by the total number of predictions per pass.

Ben Iovino 7/16/26 Biohub
"""

# %%
import matplotlib.pyplot as plt
import pandas as pd

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Trailing slash keeps "output_fold_all" from also matching "output_fold_all2".
FIRST_PASS_DIRS = ["output_fold_all", "output_fold_all2"]
SECOND_PASS_DIRS = ["output_fold_top_50"]

# %%
# TF-name lookup, restricted to the reps set
dataset = DS["tfdb4"]
rep_tfs = pd.read_csv(dataset.seq_csv)
rep_tfs["protein"] = rep_tfs["DBID"].str.lower()
name_lookup = dict(zip(rep_tfs["protein"], rep_tfs["TF_Name"]))
rep_proteins = set(rep_tfs["protein"])

# %%
# Pull every prediction with its confidence metrics and tag its pass
db = Database(job_path=f"{DATA_DIR}/jobs/hero", dataset=dataset.name)
runs_df = pd.read_sql(
    "SELECT structure_path, protein, ipTM, dom_pLDDT FROM runs", db.conn
)
db.conn.close()


def label_pass(path: str) -> str:
    """Return 'first', 'second', or None for a structure_path."""

    for d in FIRST_PASS_DIRS:
        if f"/{d}/" in path:
            return "first"
    for d in SECOND_PASS_DIRS:
        if f"/{d}/" in path:
            return "second"
    return None


runs_df["pass"] = runs_df["structure_path"].map(label_pass)
runs_df = runs_df[runs_df["pass"].notna()].copy()

# Ignore TFs that are not in the tfdb4_rep set
n_before = runs_df["protein"].nunique()
runs_df = runs_df[runs_df["protein"].isin(rep_proteins)].copy()
print(
    f"Restricted to tfdb4_rep: {runs_df['protein'].nunique()}/{n_before} proteins kept"
)
print("Structures per pass:")
print(runs_df["pass"].value_counts())

# %%
# Combined confidence: ipTM (0-1) times domain pLDDT rescaled to 0-1.
runs_df["combined"] = runs_df["ipTM"] * (runs_df["dom_pLDDT"] / 100)

# %%
# Top N% of each TF's structures (both passes pooled per TF, ranked by raw combined)
TOP_FRAC = 0.05
# Per-TF percentile rank; top 5% is the highest-ranked within each protein.
runs_df["is_top"] = (
    runs_df.groupby("protein")["combined"].rank(pct=True) >= 1 - TOP_FRAC
)

PALETTE = {"First pass": "#4C72B0", "Second pass": "#C44E52"}

# Headline: pass composition of the whole pool vs the per-TF top 1%.
comp = pd.DataFrame(
    {
        "All\nstructures": runs_df["pass"].value_counts(normalize=True) * 100,
        f"Top {TOP_FRAC:.0%}/TF\n(NA-MPNN)": runs_df.loc[
            runs_df["is_top"], "pass"
        ].value_counts(normalize=True)
        * 100,
    }
).T[["first", "second"]]
print(f"Per-TF top {TOP_FRAC:.0%}: {int(runs_df['is_top'].sum())} structures selected")
print("Pass composition (%):")
print(comp.round(1))

# Total number of predictions per pass.
counts = runs_df["pass"].value_counts().reindex(["first", "second"])
print("Predictions per pass:")
print(counts)

fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(5, 7))

# Panel A: stacked composition (first below, second above).
ax_a.bar(comp.index, comp["first"], color=PALETTE["First pass"], label="First pass")
ax_a.bar(
    comp.index,
    comp["second"],
    bottom=comp["first"],
    color=PALETTE["Second pass"],
    label="Second pass",
)
for i, col in enumerate(comp.index):
    ax_a.text(
        i,
        comp.loc[col, "first"] + comp.loc[col, "second"] / 2,
        f"{comp.loc[col, 'second']:.0f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold",
    )
ax_a.set_ylim(0, 100)
ax_a.set_ylabel("Composition (%)")
ax_a.set_xticklabels(["All", f"Top {TOP_FRAC:.0%}"], rotation=45, ha="right")
ax_a.set_title(f"Whole pool vs.\ntop {TOP_FRAC:.0%}")
ax_a.legend(title="", loc="lower center", fontsize=7)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# Panel B: total number of predictions per pass.
bar_labels = ["First pass", "Second pass"]
bars = ax_b.bar(
    bar_labels,
    counts.values,
    color=[PALETTE["First pass"], PALETTE["Second pass"]],
)
ax_b.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=10)
ax_b.set_ylabel("Number of predictions")
ax_b.tick_params(axis="x", rotation=45)
plt.setp(ax_b.get_xticklabels(), ha="right")
ax_b.set_title("Total predictions\nper pass")
ax_b.margins(y=0.12)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/passes_comp.svg", dpi=300)

# %%
