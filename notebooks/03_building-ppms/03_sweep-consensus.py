"""
Focused consensus hyperparameter sweep. A detailed num_specs x cor_threshold
sweep over the high-similarity TF set for a single, already-chosen NA-MPNN
specificity set (the winner of notebooks/03_building-ppms/03_sweep-na-mpnn.py), run for two confidence
rank metrics compared head-to-head: the historical ipTM x DBD-pLDDT
rank-percentile product versus interface-PAE alone. Produces per-rank-metric num_specs x cor
heatmaps, a rank-metric comparison grid, and cross-validates the
(num_specs, cor_threshold, rank_metric) selection to fix the deployed consensus
parameters and ranking.

Ben Iovino 8/13/26 CZ-Biohub
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af3tfm import sweep
from af3tfm.dataset import PROJECT_ROOT
from af3tfm.utils import set_plotting_style

FIG_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "tables"
TAB_DIR.mkdir(exist_ok=True)


# Determined NA-MPNN set from 03_sweep-na-mpnn.py
SPEC_SET = os.environ.get("CONSENSUS_SPEC_SET", "spec_ni30_t0.1")

# Consensus sweep axes
NUM_SPECS_GRID = sweep.num_specs_grid(
    int(os.environ.get("SWEEP_NUM_SPECS_MIN", 10)),
    int(os.environ.get("SWEEP_NUM_SPECS_MAX", 50)),
)
COR_GRID = sweep.COR_GRID

# Confidence rank metrics to compare head-to-head
RANK_COMBOS = [
    ["ipTM", "dom_pLDDT"],
    ["iface_PAE"],
]

# Hyperparameter columns swept and selected over (NA-MPNN set is fixed).
HP_COLS = ["num_specs", "cor_threshold", "rank_metric"]

# Prefix for all written outputs.
OUT_PREFIX = os.environ.get("SWEEP_OUT_PREFIX", "consensus_sweep")

# Cross-validation
RUN_CV = True
N_SPLITS = 5  # 5-fold, 80/20 train/test per fold
N_REPEATS = 5  # repeat the K-fold with fresh fold assignments (5x5 = 25 evals)
CV_SEED = 0  # base fold-assignment RNG seed (reproducible splits)
CV_CONF = 0.95  # confidence level for the interval on the CV test accuracy


def plot_rank_metric_summary(agg: pd.DataFrame, out_svg: Path) -> None:
    """
    Draw the headline rank-metric comparison: for each rank metric, the best
    value achievable over the num_specs x cor_threshold sweep.

    Args:
        agg: aggregated grid (columns num_specs, cor_threshold, rank_metric,
            sig1, sig4, mean_recovery).
        out_svg: path to write the figure to.
    """

    panels = [
        ("sig1", "Best TFs sig in top-1 cluster", "{:.0f}"),
        ("sig4", "Best TFs sig in top-4 clusters", "{:.0f}"),
        ("mean_recovery", "Best mean IC recovery", "{:.2f}"),
    ]
    panels = [p for p in panels if p[0] in agg.columns]

    # Best-achievable per rank metric, ordered by top-4 significance for a
    # stable, readable ranking across panels.
    best = agg.groupby("rank_metric").max(numeric_only=True)
    order = best.sort_values(
        "sig4" if "sig4" in best.columns else panels[0][0]
    ).index.tolist()

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (col, title, fmt) in zip(np.atleast_1d(axes).flat, panels):
        vals = [best.loc[m, col] for m in order]
        ax.barh(range(len(order)), vals, color="#4C72B0")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel(col)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)

    plt.suptitle(
        f"Rank-metric comparison: best over num_specs x cor_threshold "
        f"(NA-MPNN set {SPEC_SET})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_svg, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_per_rank_heatmaps(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Write one num_specs x cor_threshold heatmap figure per rank metric.

    Args:
        agg: aggregated grid (columns num_specs, cor_threshold, rank_metric,
            sig1, sig4, mean_recovery).
        out_dir: directory to write the figures to.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in sorted(agg["rank_metric"].unique()):
        sl = agg[agg["rank_metric"] == metric]
        if not sl.empty:
            sweep.plot_heatmaps(
                sl,
                out_dir / f"heatmaps_{metric}.svg",
                title_suffix=f"NA-MPNN set {SPEC_SET}, rank metric {metric}",
            )


def plot_from_csv() -> None:
    """
    Regenerate the rank-metric comparison grid and per-rank-metric heatmaps.
    """

    set_plotting_style()
    agg = pd.read_csv(str(TAB_DIR / f"{OUT_PREFIX}_results.csv"))
    plot_rank_metric_summary(agg, FIG_DIR / f"{OUT_PREFIX}_rank_metric_grid.svg")
    _write_per_rank_heatmaps(agg, FIG_DIR / OUT_PREFIX)


# %%
set_plotting_style()
results, jaspar_id_to_pwm, jaspar_pwm_list, job_dir = sweep.load_sweep_inputs()
print(f"Loaded {len(jaspar_id_to_pwm)} JASPAR motifs")
print(f"TFs with JASPAR match: {len(results)}")

n_cells = len(NUM_SPECS_GRID) * len(COR_GRID) * len(RANK_COMBOS)
n_workers = sweep.default_n_workers()
print(
    f"Focused sweep on {SPEC_SET}: {len(NUM_SPECS_GRID)} num_specs x "
    f"{len(COR_GRID)} cor_threshold x {len(RANK_COMBOS)} rank metrics = {n_cells} "
    f"grid cells over {len(results)} TFs on {n_workers} workers"
)

# Single fixed NA-MPNN set; no extra meta columns to merge in. The rank_metric
# axis is added by run_focused_sweep from RANK_COMBOS.
spec_sets = [{"spec_dir": job_dir / SPEC_SET, "meta": {}}]
flat = sweep.run_focused_sweep(
    results,
    jaspar_id_to_pwm,
    jaspar_pwm_list,
    job_dir,
    spec_sets,
    NUM_SPECS_GRID,
    COR_GRID,
    n_workers,
    rank_combos=RANK_COMBOS,
)

# Aggregate over the num_specs x cor_threshold x rank_metric grid.
long_df = pd.DataFrame(flat)
agg = sweep.aggregate_grid(long_df, HP_COLS)
out_csv = TAB_DIR / f"{OUT_PREFIX}_results.csv"
agg.to_csv(out_csv, index=False)

# Best achievable metric per rank metric, plus a num_specs x cor_threshold
# heatmap for each rank metric.
plot_rank_metric_summary(agg, FIG_DIR / f"{OUT_PREFIX}_rank_metric_grid.svg")
_write_per_rank_heatmaps(agg, FIG_DIR / OUT_PREFIX)

sweep.report_best_cells(agg, HP_COLS)

if RUN_CV:
    cv = sweep.run_cross_validation(
        long_df, HP_COLS, N_SPLITS, N_REPEATS, CV_SEED, sweep.CV_METRICS
    )
    cv.to_csv(str(TAB_DIR / f"{OUT_PREFIX}_cv.csv"), index=False)
    sweep.report_cross_validation(
        long_df, cv, HP_COLS, sweep.CV_METRICS, N_REPEATS, N_SPLITS, CV_CONF
    )


# %%
plot_from_csv()

# %%
