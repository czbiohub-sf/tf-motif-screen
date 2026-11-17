"""
Plot own-TF coupling and the TF-swap percentile over chromVAR SEACells.

RNA is the metacell mean of log-normalized expression; accessibility is the
chromVAR deviation z-score. All metacells are retained. The annotation is
Spearman rho and the line is a least-squares fit.
The percentile compares the same motif with pool TFs expressed in >=5% of
metacells, excluding other TFs in the own TF's family and retaining the own TF.
Correlations and percentiles are checked against the saved stage 03 results.

Load r/4.4 before running; R is used to read the existing stage 02 RDS files.
"""

# %%
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, linregress, rankdata, spearmanr

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)

RESULTS_DIR = DATA_DIR / "tools/chromvar/results/seacells"
TF_WHITELIST = DATA_DIR / "sequences/animaltfdb/tfdb4_tf_whitelist.txt"
TF_FAMILIES = DATA_DIR / "sequences/animaltfdb/tfdb4_tfs.csv"
RESOURCE_LABELS = {"zebramotif": "ZebraMotif", "cisbp": "CisBP"}
RESOURCE_COLORS = {"zebramotif": "#2166ac", "cisbp": "#E2A32D"}
FIGURE_WIDTH_SCALE = 0.8

# HYPERPARAMETERS
TF = "otx2b"
SOURCE = "zebramotif"  # Or "cisbp".


# %%
# Extract the saved metacell expression and motif deviations
# Keep RDS reading in R so sparse matrices and Bioconductor classes retain their
# native definitions. Arguments are passed directly, without shell interpolation.
R_EXTRACT = r"""
suppressPackageStartupMessages({
    library(Matrix)
    library(SummarizedExperiment)
    library(chromVAR)
})
args <- commandArgs(trailingOnly = TRUE)
results_dir <- args[1]; tf <- args[2]; motif_id <- args[3]
whitelist <- tolower(trimws(readLines(args[4])))
output_dir <- args[5]
mcs <- readRDS(file.path(results_dir, "metacells.rds"))
stopifnot(tf %in% rownames(mcs$rna_mc))
expr <- setNames(as.numeric(mcs$rna_mc[tf, ]), colnames(mcs$rna_mc))
pool <- rownames(mcs$rna_mc)[tolower(rownames(mcs$rna_mc)) %in% whitelist]
tf_expression <- as.matrix(mcs$rna_mc[pool, , drop = FALSE])
celltype <- mcs$mc_celltype
rm(mcs)
invisible(gc())
dev <- readRDS(file.path(results_dir, "chromvar_deviations.rds"))
z <- assay(dev, "z")
ids <- colnames(z)
stopifnot(motif_id %in% rownames(z), !anyDuplicated(ids),
          !anyDuplicated(names(expr)), setequal(ids, names(expr)),
          all(ids %in% names(celltype)))
points <- data.frame(
    metacell = ids,
    expression = unname(expr[ids]),
    deviation_z = as.numeric(z[motif_id, ids]),
    celltype = as.character(celltype[ids])
)
write.table(points, file.path(output_dir, "points.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(tf = rownames(tf_expression),
                       tf_expression[, ids, drop = FALSE], check.names = FALSE),
            file.path(output_dir, "tf_expression.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
"""


def extract_data(
    results_dir: Path, tf: str, motif_id: str, whitelist: Path | None, families: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract aligned motif deviations and pool TF expression from RDS."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required; run `module load r/4.4` first.")
    print(f"Reading {tf} / {motif_id} from {results_dir}", flush=True)
    with TemporaryDirectory(prefix="chromvar_own_tf_") as temporary:
        if whitelist is None:
            whitelist = Path(temporary) / "tf_universe.txt"
            universe = pd.read_csv(families, usecols=["TF_Name"])["TF_Name"].dropna()
            whitelist.write_text("\n".join(universe.str.lower()) + "\n")
            print(
                "Using the TF-family table as the TF universe (R fallback).", flush=True
            )
        subprocess.run(
            [
                rscript,
                "--vanilla",
                "-e",
                R_EXTRACT,
                str(results_dir),
                tf,
                motif_id,
                str(whitelist),
                temporary,
            ],
            check=True,
        )
        points = pd.read_csv(Path(temporary) / "points.tsv", sep="\t")
        tf_expression = pd.read_csv(
            Path(temporary) / "tf_expression.tsv", sep="\t", index_col="tf"
        )
    if len(points) < 3 or not points["metacell"].is_unique:
        raise ValueError("Expected at least three uniquely identified metacells.")
    values = points[["expression", "deviation_z"]].to_numpy()
    if not np.isfinite(values).all() or np.any(np.ptp(values, axis=0) == 0):
        raise ValueError("Expression and deviations must be finite and nonconstant.")
    if (
        not tf_expression.index.is_unique
        or not tf_expression.columns.equals(pd.Index(points["metacell"]))
        or not np.isfinite(tf_expression.to_numpy()).all()
    ):
        raise ValueError("TF expression must be finite and aligned to the metacells.")
    return points, tf_expression


# %%
# Own-TF coupling plot and statistics
def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    for extension in ("svg", "png"):
        path = figure_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()
    plt.close(fig)


def plot_coupling(points: pd.DataFrame, row: pd.Series, figure_dir: Path) -> dict:
    """Plot every metacell, verifying rho against the saved pipeline result."""
    x = points["expression"].to_numpy()
    y = points["deviation_z"].to_numpy()
    rho = float(spearmanr(x, y).statistic)
    if not np.isclose(rho, row["rho"], atol=1e-10, rtol=1e-8):
        raise ValueError(f"Recomputed rho {rho} differs from saved rho {row['rho']}.")
    fit = linregress(x, y)

    width, height = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width * FIGURE_WIDTH_SCALE, height))
    ax.scatter(x, y, color=RESOURCE_COLORS[row["source"]], alpha=0.55, linewidths=0)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, fit.intercept + fit.slope * xx, color="#c0392b")
    ax.text(
        0.04,
        0.96,
        f"Spearman $\\rho$ = {rho:.3f}\n$n$ = {len(points):,} SEACells",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3},
    )
    ax.set_xlabel(f"{row['tf']} expression\n(mean log-normalized RNA)")
    ax.set_ylabel("Own-motif deviation\n(chromVAR z-score)")
    ax.set_title(row["tf"], loc="left", fontstyle="italic")
    ax.set_title(RESOURCE_LABELS[row["source"]], loc="right", color="0.4")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.04, y=0.08)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(0.18, 0.025, f"{row['motif_id']}  |  line: least-squares fit", color="0.4")
    stem = f"chromvar_own_tf_coupling_{row['tf']}_{row['source']}"
    save_figure(fig, figure_dir, stem)
    return {
        "tf": row["tf"],
        "motif_id": row["motif_id"],
        "source": row["source"],
        "n_metacells": len(points),
        "spearman_rho": rho,
        "pipeline_rho": row["rho"],
        "own_pctile": row["own_pctile"],
        "linear_slope": fit.slope,
        "linear_intercept": fit.intercept,
        "pearson_r": fit.rvalue,
    }


# %%
# TF-swap correlations and the own-TF percentile
def percentile_correlations(
    tf_expression: pd.DataFrame,
    points: pd.DataFrame,
    row: pd.Series,
    families_path: Path,
) -> pd.DataFrame:
    """Reproduce make_own_percentile()'s TF pool and average-rank correlations."""
    expressed_lc = tf_expression.index.str.lower()[
        (tf_expression > 0).mean(axis=1) >= 0.05
    ]
    # Match R's named-vector lookup: each lowercase symbol maps to its first RNA row.
    expression = tf_expression.loc[~tf_expression.index.str.lower().duplicated()]
    expression = expression.loc[expression.index.str.lower().isin(expressed_lc)]
    families = pd.read_csv(families_path, usecols=["TF_Name", "Family"])
    families["TF_Name"] = families["TF_Name"].str.lower()
    families["Family"] = families["Family"].str.lower()
    family_map = families.drop_duplicates("TF_Name").set_index("TF_Name")["Family"]

    ranks = rankdata(expression.to_numpy(), axis=1, method="average")
    ranks -= ranks.mean(axis=1, keepdims=True)
    # The R implementation assigns constant expression rows a zero correlation.
    ranks /= np.sqrt(
        np.maximum((ranks**2).sum(axis=1, keepdims=True), np.finfo(float).eps)
    )
    deviation_rank = rankdata(points["deviation_z"].to_numpy(), method="average")
    deviation_rank -= deviation_rank.mean()
    deviation_rank /= np.linalg.norm(deviation_rank)

    own_tf = row["tf"].lower()
    result = pd.DataFrame(
        {
            "pool_tf": expression.index,
            "family": family_map.reindex(expression.index.str.lower()).to_numpy(),
            "rho": ranks @ deviation_rank,
            "is_own_tf": expression.index.str.lower() == own_tf,
        }
    )
    own_family = family_map.get(own_tf)
    if pd.notna(own_family):
        result = result.loc[result["family"].ne(own_family) | result["is_own_tf"]]
    result = result.reset_index(drop=True)
    own = result.loc[result["is_own_tf"], "rho"]
    if len(own) != 1 or not np.isclose(own.iloc[0], row["rho"], atol=1e-10, rtol=1e-8):
        raise ValueError(
            "Own-TF rank correlation does not match the saved pipeline result."
        )
    percentile = result["rho"].le(own.iloc[0]).mean()
    if not np.isclose(percentile, row["own_pctile"], atol=1e-12, rtol=0):
        raise ValueError(
            f"Recomputed percentile {percentile} differs from saved {row['own_pctile']}."
        )
    return result


def plot_percentile(
    correlations: pd.DataFrame, row: pd.Series, n_metacells: int, figure_dir: Path
) -> dict:
    """Histogram/density of motif-vs-TF correlations with the own TF marked."""
    rho = correlations["rho"].to_numpy()
    own_rho = correlations.loc[correlations["is_own_tf"], "rho"].iloc[0]
    percentile = float(np.mean(rho <= own_rho))
    # Match R density()'s default bw.nrd0 with adjust=1.1, as in the R plot.
    sd = rho.std(ddof=1)
    iqr = np.subtract(*np.percentile(rho, [75, 25]))
    scale = min(sd, iqr / 1.34) or sd
    bandwidth = 0.9 * scale * len(rho) ** (-0.2) * 1.1
    density = gaussian_kde(rho, bw_method=bandwidth / sd)
    xx = np.linspace(rho.min() - 3 * bandwidth, rho.max() + 3 * bandwidth, 512)

    color = RESOURCE_COLORS[row["source"]]
    width, height = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(width * FIGURE_WIDTH_SCALE, height))
    ax.hist(rho, bins=60, density=True, color=color, alpha=0.55, edgecolor="none")
    ax.plot(xx, density(xx), color=color)
    ax.axvline(0, linestyle=":", color="0.55")
    ax.axvline(own_rho, color="#c0392b")
    ax.annotate(
        f"{row['tf']}: Spearman $\\rho$ = {own_rho:.3f}\n"
        f"{percentile * 100:.2f}th percentile of {len(rho):,} TFs",
        xy=(own_rho, 0.96),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        color="#c0392b",
    )
    ax.set_xlabel(
        f"Spearman correlation of {row['tf']} motif accessibility\n"
        "with each pool TF's expression"
    )
    ax.set_ylabel("Density")
    ax.set_title(f"Own-TF percentile: {row['tf']}", loc="left")
    ax.set_title(RESOURCE_LABELS[row["source"]], loc="right", color="0.4")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.text(
        0.5,
        0.025,
        f"{row['motif_id']}  |  {n_metacells:,} SEACells; "
        "same-family TFs excluded, own TF retained",
        ha="center",
        color="0.4",
    )
    save_figure(
        fig, figure_dir, f"chromvar_own_tf_percentile_{row['tf']}_{row['source']}"
    )
    return {
        "tf": row["tf"],
        "motif_id": row["motif_id"],
        "source": row["source"],
        "n_metacells": n_metacells,
        "n_pool_tfs_including_own": len(rho),
        "own_rho": own_rho,
        "own_pctile": percentile,
        "pipeline_own_pctile": row["own_pctile"],
        "n_pool_correlations_le_own": int(np.sum(rho <= own_rho)),
        "density_bandwidth": bandwidth,
    }


# %%
# Select the primary motif for the requested TF and resource
if SOURCE not in RESOURCE_LABELS:
    raise ValueError(f"SOURCE must be one of {tuple(RESOURCE_LABELS)}.")

correlations = pd.read_csv(RESULTS_DIR / "correlations.tsv", sep="\t")
selected = correlations.loc[
    correlations["tf"].str.lower().eq(TF.strip().lower())
    & correlations["source"].eq(SOURCE)
]
if len(selected) != 1:
    raise ValueError(f"Expected one primary {SOURCE} motif for {TF}.")
row = selected.iloc[0]

# %%
# Load aligned expression and accessibility for every SEACell
# Fall back to the TF-family table when the whitelist is absent, as in R.
whitelist = (
    TF_WHITELIST if TF_WHITELIST is not None and TF_WHITELIST.is_file() else None
)
points, tf_expression = extract_data(
    RESULTS_DIR, row["tf"], row["motif_id"], whitelist, TF_FAMILIES
)
pool_correlations = percentile_correlations(tf_expression, points, row, TF_FAMILIES)
points.insert(1, "tf", row["tf"])
points.insert(2, "motif_id", row["motif_id"])
points.insert(3, "source", row["source"])

# %%
# Plot own-motif accessibility against TF expression
summary = plot_coupling(points, row, FIG_DIR)
print(
    f"{row['tf']}: n = {len(points):,}, Spearman rho = {summary['spearman_rho']:.6f} "
    f"(matches stage 03); own-TF percentile = {100 * row['own_pctile']:.2f}%"
)

# %%
# Place the own-TF correlation within the TF-swap distribution
percentile_summary = plot_percentile(pool_correlations, row, len(points), FIG_DIR)
percentile_summary["tf_universe_file"] = str(whitelist or TF_FAMILIES)
print(
    f"TF-swap plot: {len(pool_correlations):,} pool TFs; "
    f"percentile = {percentile_summary['own_pctile']:.12f} (matches stage 03)."
)

# %%
# Save the plotted values and statistics
stem = f"chromvar_own_tf_coupling_{row['tf']}_{row['source']}"
points.to_csv(TAB_DIR / f"{stem}_points.tsv", sep="\t", index=False)
pd.DataFrame([summary]).to_csv(TAB_DIR / f"{stem}_stats.tsv", sep="\t", index=False)

percentile_stem = f"chromvar_own_tf_percentile_{row['tf']}_{row['source']}"
pool_correlations.to_csv(
    TAB_DIR / f"{percentile_stem}_correlations.tsv", sep="\t", index=False
)
pd.DataFrame([percentile_summary]).to_csv(
    TAB_DIR / f"{percentile_stem}_stats.tsv", sep="\t", index=False
)
