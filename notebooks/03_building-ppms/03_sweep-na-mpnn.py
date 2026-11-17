"""
Broad NA-MPNN parameter sweep. For each (number_of_batches, temperature) cell,
run the focused num_specs x cor_threshold mini-sweep over the high-similarity TF
set and record the best achievable JASPAR-significance / IC-recovery. The
num_specs x cor sweep is run independently over each NA-MPNN set, so the two
NA-MPNN knobs become two more axes of one 4-D hyperparameter grid. Use this to
pick the NA-MPNN operating point to carry into the focused consensus sweep
(notebooks/03_building-ppms/03_sweep-consensus.py).

Ben Iovino 8/13/26 Biohub
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


# Focused mini-sweep axes. Extents are env-overridable so an alternate sweep
# (e.g. a single NA-MPNN set explored more deeply) can reuse this unchanged.
NUM_SPECS_GRID = sweep.num_specs_grid(
    int(os.environ.get("SWEEP_NUM_SPECS_MIN", 10)),
    int(os.environ.get("SWEEP_NUM_SPECS_MAX", 30)),
)
COR_GRID = sweep.COR_GRID

# NA-MPNN parameter sets: one specificity directory per cell under data/jobs/hero,
# produced by scripts/run_spec_nampnn_sweep.sh. SWEEP_SINGLE_SET="spec_ni30_t0.1:30:0.1"
# restricts the broad sweep to one set for a deeper single-set exploration.
_SINGLE_SET = os.environ.get("SWEEP_SINGLE_SET")  # "spec_dir:num_inferences:temp"
if _SINGLE_SET:
    _name, _ni, _t = _SINGLE_SET.split(":")
    NAMPNN_GRID = [(_name, int(_ni), float(_t))]
else:
    NAMPNN_GRID = [
        (f"spec_ni{ni}_t{t}", ni, t)
        for ni in (1, 5, 10, 20, 30)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5)
    ]

# Hyperparameter columns swept and selected over
HP_COLS = ["num_specs", "cor_threshold", "num_inferences", "temperature"]

# Prefix for all written outputs so alternate sweeps don't clobber the default.
OUT_PREFIX = os.environ.get("SWEEP_OUT_PREFIX", "nampnn_sweep")

# Cross-validation
RUN_CV = True
N_SPLITS = 5  # 5-fold, 80/20 train/test per fold
N_REPEATS = 5  # repeat the K-fold with fresh fold assignments (5x5 = 25 evals)
CV_SEED = 0  # base fold-assignment RNG seed (reproducible splits)
CV_CONF = 0.95  # confidence level for the interval on the CV test accuracy


def plot_nampnn_summary(agg: pd.DataFrame, out_svg: Path) -> None:
    """
    Draw the headline NA-MPNN grid: for each (num_inferences, temperature) cell,
    the best value achievable over the num_specs x cor_threshold sweep.

    Args:
        agg: full 4-D aggregated grid (columns num_specs, cor_threshold,
            num_inferences, temperature, sig1, sig4, mean_recovery).
        out_svg: path to write the figure to.
    """

    ni_vals = sorted(agg["num_inferences"].unique())
    t_vals = sorted(agg["temperature"].unique())

    panels = [
        ("sig1", "Best TFs sig in top-1 cluster", "viridis", "{:.0f}"),
        ("sig4", "Best TFs sig in top-4 clusters", "viridis", "{:.0f}"),
        ("mean_recovery", "Best mean IC recovery", "magma", "{:.2f}"),
    ]
    panels = [p for p in panels if p[0] in agg.columns]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    for ax, (col, title, cmap, fmt) in zip(np.atleast_1d(axes).flat, panels):
        grid = np.full((len(ni_vals), len(t_vals)), np.nan)
        for i, ni in enumerate(ni_vals):
            for j, t in enumerate(t_vals):
                sub = agg[
                    (agg["num_inferences"] == ni) & np.isclose(agg["temperature"], t)
                ]
                if not sub.empty:
                    grid[i, j] = sub[col].max()
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(t_vals)))
        ax.set_xticklabels([f"{t:.1f}" for t in t_vals])
        ax.set_yticks(range(len(ni_vals)))
        ax.set_yticklabels(ni_vals)
        ax.set_xlabel("temperature")
        ax.set_ylabel("num_inferences (batches)")
        ax.set_title(title)
        for i in range(len(ni_vals)):
            for j in range(len(t_vals)):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        fmt.format(grid[i, j]),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=9,
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "NA-MPNN parameter sweep: best over num_specs x cor_threshold",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_svg, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_per_set_heatmaps(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Write one num_specs x cor_threshold heatmap figure per NA-MPNN set.

    Args:
        agg: full 4-D aggregated grid (columns num_specs, cor_threshold,
            num_inferences, temperature, sig1, sig4, mean_recovery).
        out_dir: directory to write the figures to.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    for _, ni, temp in NAMPNN_GRID:
        sl = agg[(agg["num_inferences"] == ni) & np.isclose(agg["temperature"], temp)]
        if not sl.empty:
            sweep.plot_heatmaps(
                sl,
                out_dir / f"heatmaps_ni{ni}_t{temp}.svg",
                title_suffix=f"num_inferences={ni}, temperature={temp}",
            )


def plot_from_csv() -> None:
    """
    Regenerate the NA-MPNN summary grid and per-set heatmaps from the CSV.
    """

    set_plotting_style()
    agg = pd.read_csv(str(TAB_DIR / f"{OUT_PREFIX}_results.csv"))
    plot_nampnn_summary(agg, FIG_DIR / f"{OUT_PREFIX}_nampnn_grid.svg")
    _write_per_set_heatmaps(agg, FIG_DIR / OUT_PREFIX)


# %%
set_plotting_style()
results, jaspar_id_to_pwm, jaspar_pwm_list, job_dir = sweep.load_sweep_inputs()
print(f"Loaded {len(jaspar_id_to_pwm)} JASPAR motifs")
print(f"TFs with JASPAR match: {len(results)}")

n_cells = len(NUM_SPECS_GRID) * len(COR_GRID) * len(NAMPNN_GRID)
n_workers = sweep.default_n_workers()
print(
    f"Sweeping {len(NUM_SPECS_GRID)} num_specs x {len(COR_GRID)} cor_threshold x "
    f"{len(NAMPNN_GRID)} NA-MPNN sets = {n_cells} grid cells over {len(results)} TFs "
    f"on {n_workers} workers"
)

# One specificity directory per NA-MPNN set; meta columns (ni, temp) are merged
# into every record so the two NA-MPNN knobs become extra grid axes.
spec_sets = [
    {"spec_dir": job_dir / name, "meta": {"num_inferences": ni, "temperature": temp}}
    for name, ni, temp in NAMPNN_GRID
]
flat = sweep.run_focused_sweep(
    results,
    jaspar_id_to_pwm,
    jaspar_pwm_list,
    job_dir,
    spec_sets,
    NUM_SPECS_GRID,
    COR_GRID,
    n_workers,
)

# Aggregate over the full 4-D hyperparameter grid.
long_df = pd.DataFrame(flat)
agg = sweep.aggregate_grid(long_df, HP_COLS)
out_csv = TAB_DIR / f"{OUT_PREFIX}_results.csv"
agg.to_csv(out_csv, index=False)

# Best achievable metric per NA-MPNN (num_inferences x temperature) with a
# num_specs x cor_threshold heatmap for each set.
plot_nampnn_summary(agg, FIG_DIR / f"{OUT_PREFIX}_nampnn_grid.svg")
_write_per_set_heatmaps(agg, FIG_DIR / OUT_PREFIX)

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
