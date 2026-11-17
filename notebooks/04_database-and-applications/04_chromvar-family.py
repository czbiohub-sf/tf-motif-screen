"""
Compare chromVAR own-TF percentiles across DNA-binding-domain (DBD) classes.

Classes are selected by summed motif coverage across ZebraMotif, JASPAR, and
CisBP using notebooks/dbd_crosswalk.yaml. Distributions show median crossbars,
tested-TF counts, and selected ZebraMotif TFs; small groups use individual points.
Coverage comes from the current database export and percentiles from saved
chromVAR correlations. Other/unclassified classes are excluded.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)

RESULTS_DIR = DATA_DIR / "tools/chromvar/results/seacells"
CROSSWALK = PROJECT_ROOT / "notebooks/dbd_crosswalk.yaml"
TF_FAMILIES = DATA_DIR / "sequences/animaltfdb/tfdb4_tfs.csv"
PWM_MODES = DATA_DIR / "database/export/database/pwm_modes.parquet"
JASPAR = PROJECT_ROOT / "notebooks/jaspar_core.tsv"
CISBP = DATA_DIR / "sequences/cisbp/TF_Information.txt"
SOURCES = ("cisbp", "zebramotif")
LABELS = {"cisbp": "CisBP", "zebramotif": "ZebraMotif"}
COLORS = {"cisbp": "#E2A32D", "zebramotif": "#2166ac"}

# HYPERPARAMETERS
TOP_N = 15
SORT_BY = "coverage"  # Or "median-difference": ZebraMotif minus CisBP median.
HIGHLIGHTS = ("otx2b", "tbx6", "her7", "drll.3", "rreb1b")
MIN_VIOLIN = 10


# %%
# Map native family names to canonical DBD classes
def native_map(crosswalk: dict, field: str, lowercase: bool = False) -> dict:
    """Match R's first-match lookup for native family names."""
    result = {}
    for canonical, fields in crosswalk.items():
        for term in fields.get(field, []):
            result.setdefault(term.lower() if lowercase else term, canonical)
    return result


def family_coverage(crosswalk: dict, pwm_modes: Path) -> pd.DataFrame:
    """Reproduce the R plot's class ranking from the three coverage resources."""
    animal = native_map(crosswalk, "animaltfdb", lowercase=True)
    modes = pd.read_parquet(pwm_modes, columns=["DBID", "Family"])
    modes = modes.loc[~modes["DBID"].astype(str).str.contains(r"_\d+$")]
    modes = modes.drop_duplicates("DBID")
    modes["family"] = modes["Family"].str.lower().map(animal)
    zm_counts = modes.groupby("family")["DBID"].nunique(dropna=False)

    jaspar = pd.read_csv(JASPAR, sep="\t", na_values=["NA", ""])
    jaspar = jaspar.loc[jaspar["tax_group"].eq("vertebrates")].drop_duplicates(
        "base_id"
    )
    family = jaspar["family"].map(native_map(crosswalk, "jaspar_family"))
    base_class = jaspar["class"].str.split("::").str[0]
    jaspar["canonical"] = family.fillna(
        base_class.map(native_map(crosswalk, "jaspar_class"))
    )
    jaspar["tf_key"] = jaspar["uniprot_ids"].fillna(jaspar["name"])
    jas_counts = jaspar.groupby("canonical")["tf_key"].nunique(dropna=False)

    cisbp = pd.read_csv(CISBP, sep="\t")
    cisbp["family"] = cisbp["Family_Name"].map(native_map(crosswalk, "cisbp"))
    cis_counts = cisbp.groupby("family")["DBID"].nunique(dropna=False)

    coverage = pd.DataFrame(index=pd.Index(crosswalk, name="family"))
    for name, counts in (
        ("zebramotif", zm_counts),
        ("jaspar", jas_counts),
        ("cisbp", cis_counts),
    ):
        coverage[name] = counts.reindex(coverage.index).fillna(0).astype(int)
    coverage["total"] = coverage.sum(axis=1)
    coverage = coverage.drop(index=["Other", "Other / unclassified"], errors="ignore")
    # Stable sorting retains crosswalk order for tied coverage totals.
    return coverage.sort_values("total", ascending=False, kind="stable").reset_index()


def load_percentiles(path: Path, crosswalk: dict, families: list[str]) -> pd.DataFrame:
    annotations = pd.read_csv(TF_FAMILIES, usecols=["TF_Name", "Family"])
    annotations["tf"] = annotations["TF_Name"].str.lower()
    annotations["family"] = (
        annotations["Family"]
        .str.lower()
        .map(native_map(crosswalk, "animaltfdb", lowercase=True))
    )
    # Match R's named-vector lookup when gene symbols have case variants.
    lookup = annotations.drop_duplicates("tf").set_index("tf")["family"]
    points = pd.read_csv(path, sep="\t")
    points = points.loc[
        points["source"].isin(SOURCES) & points["own_pctile"].notna()
    ].copy()
    points["family"] = points["tf"].str.lower().map(lookup)
    points = points.loc[points["family"].isin(families)].copy()
    if points.duplicated(["source", "tf"]).any():
        raise ValueError(
            "Expected one primary motif per TF and resource in correlations.tsv."
        )
    if (
        not np.isfinite(points["own_pctile"]).all()
        or not points["own_pctile"].between(0, 1).all()
    ):
        raise ValueError("Own-TF percentiles must be finite and between zero and one.")
    return points


def families_by_median_difference(
    points: pd.DataFrame, families: list[str]
) -> list[str]:
    """Order selected classes by ZebraMotif's median advantage over CisBP."""
    medians = (
        points.groupby(["family", "source"])["own_pctile"].median().unstack("source")
    )
    medians = medians.reindex(index=families, columns=SOURCES)
    difference = medians["zebramotif"] - medians["cisbp"]
    return difference.sort_values(
        ascending=False, kind="stable", na_position="last"
    ).index.tolist()


def family_summary(
    points: pd.DataFrame, coverage: pd.DataFrame, families: list[str]
) -> pd.DataFrame:
    summary = (
        points.groupby(["family", "source"])
        .agg(
            n_tested=("tf", "size"),
            median_pctile=("own_pctile", "median"),
            frac_top5=("own_pctile", lambda values: values.ge(0.95).mean()),
        )
        .reset_index()
    )
    summary[["median_pctile", "frac_top5"]] = summary[
        ["median_pctile", "frac_top5"]
    ].round(3)
    summary = summary.merge(coverage, on="family", validate="many_to_one")
    summary["family"] = pd.Categorical(
        summary["family"], categories=families, ordered=True
    )
    summary["source"] = summary["source"].map(LABELS)
    return summary.sort_values(["family", "source"], kind="stable")


# %%
# Family distributions, median crossbars, and highlighted TFs
def plot_family(
    points: pd.DataFrame, families: list[str], figure_dir: Path, highlights: list[str]
) -> None:
    fig, ax = plt.subplots(figsize=(6 * 0.8, 6))
    ax.axvline(0.5, color="0.4", linestyle=":", zorder=0)
    for index, family in enumerate(families):
        subset = points.loc[points["family"].eq(family)]
        both = subset["source"].nunique() == 2
        half_width = 0.225 if both else 0.45
        highlight_rows = subset.loc[
            subset["source"].eq("zebramotif")
            & subset["tf"].str.lower().isin(highlights)
        ]
        for source in SOURCES:
            values = subset.loc[subset["source"].eq(source), "own_pctile"].to_numpy()
            if not len(values):
                continue
            position = index + (
                {"cisbp": 0.2, "zebramotif": -0.2}[source] if both else 0
            )
            color = COLORS[source]
            if len(values) >= MIN_VIOLIN and np.ptp(values) > 0:
                # Same bandwidth rule as geom_violin(): bw.nrd0, adjust=1.
                sd = values.std(ddof=1)
                iqr = np.subtract(*np.percentile(values, [75, 25]))
                bandwidth = 0.9 * (min(sd, iqr / 1.34) or sd) * len(values) ** (-0.2)
                kde = gaussian_kde(values, bw_method=bandwidth / sd)
                xx = np.linspace(values.min(), values.max(), 512)
                density = kde(xx)
                extent = density / density.max() * half_width
                ax.fill_between(
                    xx,
                    position - extent,
                    position + extent,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.5,
                    linewidth=0.8,
                )
            else:
                ax.scatter(
                    values, np.full(len(values), position), color=color, s=9, alpha=0.8
                )
            median_half_width = 0.12 if both else 0.25
            ax.vlines(
                np.median(values),
                position - median_half_width,
                position + median_half_width,
                color="0.15",
                linewidth=1.5,
            )
            ax.text(
                values.max() + 0.02,
                position,
                str(len(values)),
                va="center",
                fontsize=8,
                color="0.3",
            )
        for order, (_, row) in enumerate(highlight_rows.iterrows()):
            position = index - 0.2 if both else index
            ax.scatter(
                row["own_pctile"],
                position,
                marker="D",
                s=24,
                color="0.08",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            ax.annotate(
                row["tf"],
                (row["own_pctile"], position),
                xytext=(-5, (order - (len(highlight_rows) - 1) / 2) * 10),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
    ax.set_yticks(
        range(len(families)),
        families,
        rotation=45,
        ha="right",
        va="center",
        rotation_mode="anchor",
    )
    ax.set_ylim(len(families) - 0.4, -0.6)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Own-TF percentile")
    ax.set_ylabel("DBD class")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(
                facecolor=COLORS[source],
                edgecolor=COLORS[source],
                alpha=0.5,
                label=LABELS[source],
            )
            for source in SOURCES
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=1,
        handlelength=1.2,
    )
    fig.suptitle("Own-TF coupling by DBD class", y=0.99)
    fig.text(
        0.5,
        0.015,
        "Median crossbars; dotted line = 0.5 null",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    for extension in ("svg", "png"):
        path = figure_dir / f"percentile_family.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()
    plt.close(fig)


# %%
# Select the top classes by motif coverage
if TOP_N < 1:
    raise ValueError("TOP_N must be positive.")
if SORT_BY not in ("coverage", "median-difference"):
    raise ValueError("SORT_BY must be 'coverage' or 'median-difference'.")

crosswalk = yaml.safe_load(CROSSWALK.read_text())
coverage = family_coverage(crosswalk, PWM_MODES)
families = coverage.head(TOP_N)["family"].tolist()

# %%
# Load own-TF percentiles and order the selected classes
points = load_percentiles(RESULTS_DIR / "correlations.tsv", crosswalk, families)
if SORT_BY == "median-difference":
    families = families_by_median_difference(points, families)
summary = family_summary(points, coverage, families)
print(f"Family order ({SORT_BY}):", ", ".join(families))

# %%
# Compare CisBP and ZebraMotif within each DBD class
plot_family(points, families, FIG_DIR, [tf.lower() for tf in HIGHLIGHTS])

# %%
# Save the plotted values, family statistics, and coverage counts
points.to_csv(TAB_DIR / "percentile_family_points.tsv", sep="\t", index=False)
summary.to_csv(TAB_DIR / "percentile_family.tsv", sep="\t", index=False)
coverage.to_csv(TAB_DIR / "percentile_family_coverage.tsv", sep="\t", index=False)
print(f"Plotted {len(points):,} resource/TF pairs across {len(families)} DBD classes.")
