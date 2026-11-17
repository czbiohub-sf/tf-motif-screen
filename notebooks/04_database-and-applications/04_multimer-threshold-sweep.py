"""
Threshold-robustness sweep for the multimer specificity analyses.

For each co-folded complex (heterodimer myod1/tcf3a, homodimers tp63 and tp73,
heterotetramer tp63/tp73) this sweeps the consensus clustering hyperparameters
(num_specs x cor_threshold) and maps two *reference-blind* properties of the
primary (largest) cluster:

  - member_frac : primary-cluster size / num_specs (how dominant the mode is)
  - stability   : bootstrap reproducibility of the primary-mode consensus

and, overlaid only as an independent readout, the JASPAR similarity of that
primary consensus. The database operating point and the alternative multimer
point are marked on every map.

The argument this figure supports (see the paper methods): the multimer primary
mode is a broad, reproducible basin, and its clustering parameters are chosen by
these reference-blind stability criteria -- NOT by agreement with JASPAR. The
JASPAR panel is the consequence, not the objective.

Ben Iovino 8/26/26 Biohub
"""

# %%
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import load_homer_multi_motif_file
from af3tfm.multimer_sweep import rank_and_load, sweep_condition
from af3tfm.utils import set_plotting_style

DATA = Path(DATA_DIR)
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)

# Sweep axes. num_specs is capped per condition by its available pool.
MAX_SPECS = 60
NUM_SPECS_GRID = list(range(10, MAX_SPECS + 1, 2))
COR_GRID = [round(0.45 + 0.05 * i, 2) for i in range(10)]  # 0.45 .. 0.90
N_BOOT = 30

# Operating points to mark on every map: (num_specs, cor_threshold, label).
# DB point = deployed database consensus; ALT = the lower-specs/lower-cor point
# that gives the cleaner multimer primary modes.
DB_POINT = (45, 0.77, "DB (45/0.77)")
ALT_POINT = (25, 0.60, "alt (25/0.60)")

HERO = DATA / "jobs" / "hero"


# %%
# JASPAR references (also the empirical null for the p-value readout)
jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
jaspar_pwm_list = list(jaspar_id_to_pwm.values())
print(f"Loaded {len(jaspar_id_to_pwm)} JASPAR motifs")


# %%
# Define the four multimer conditions. Each entry loads a rank-ordered PPM pool
# (one PPM per distinct DNA probe) and names the JASPAR reference to read out.
def _load_runs(job: str, protein: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(DATA / "jobs" / job / "jobdb.sqlite"))
    runs = pd.read_sql("SELECT * FROM runs WHERE protein = ?", conn, params=(protein,))
    conn.close()
    return runs


MYOD1, TCF3A = "ensdarg00000030110", "ensdarg00000005915"
TP63, TP73 = "ensdarg00000113627", "ensdarg00000017953"

CONDITIONS = []

# Heterodimer myod1+tcf3a: co-folded npz keyed by both proteins; read out vs each ref.
hetero_dir = (
    DATA / "jobs" / "dimers-myod1-tcf3a" / "specificities" / MYOD1 / "specificity"
)
hetero_runs = _load_runs("dimers-myod1-tcf3a", MYOD1)
for ref_tf, ref_id in [("myod1", "MA0499.3"), ("tcf3a", "MA0522.4")]:
    CONDITIONS.append(
        {
            "name": f"heterodimer_vs_{ref_tf}",
            "title": f"myod1/tcf3a heterodimer vs JASPAR {ref_id} ({ref_tf})",
            "runs": hetero_runs,
            "spec_dir": hetero_dir,
            "name_fn": lambda r: f"{MYOD1}_{TCF3A}_{r.dna}_{r.seed}.npz",
            "ref_id": ref_id,
        }
    )

# Homodimers tp63 and tp73.
for tf, prot, job, ref_id in [
    ("tp63", TP63, "dimers-tp63", "MA0525.2"),
    ("tp73", TP73, "dimers-tp73", "MA0861.2"),
]:
    CONDITIONS.append(
        {
            "name": f"homodimer_{tf}",
            "title": f"{tf} homodimer vs JASPAR {ref_id}",
            "runs": _load_runs(job, prot),
            "spec_dir": DATA / "jobs" / job / "specificities" / prot / "specificity",
            "name_fn": (lambda p: (lambda r: f"{p}_{r.motif}_{r.dna}_{r.seed}.npz"))(
                prot
            ),
            "ref_id": ref_id,
        }
    )

# Heterotetramer tp63/tp73: read out vs both references.
tet_dir = DATA / "jobs" / "tetramer-tp63-tp73" / "specificities" / TP63 / "specificity"
tet_runs = _load_runs("tetramer-tp63-tp73", TP63)
for ref_tf, ref_id in [("tp63", "MA0525.2"), ("tp73", "MA0861.2")]:
    CONDITIONS.append(
        {
            "name": f"tetramer_vs_{ref_tf}",
            "title": f"tp63/tp73 tetramer vs JASPAR {ref_id} ({ref_tf})",
            "runs": tet_runs,
            "spec_dir": tet_dir,
            "name_fn": (lambda p: (lambda r: f"{p}_{r.motif}_{r.dna}_{r.seed}.npz"))(
                TP63
            ),
            "ref_id": ref_id,
        }
    )


# %%
# Run the sweep for every condition.
grids = {}
for cond in CONDITIONS:
    loaded = rank_and_load(cond["runs"], cond["spec_dir"], cond["name_fn"], MAX_SPECS)
    ns_grid = [n for n in NUM_SPECS_GRID if n <= len(loaded)]
    print(
        f"[{cond['name']}] pool={len(loaded)}  num_specs up to {ns_grid[-1] if ns_grid else 0}"
    )
    grid = sweep_condition(
        loaded,
        jaspar_id_to_pwm[cond["ref_id"]],
        jaspar_pwm_list,
        ns_grid,
        COR_GRID,
        n_boot=N_BOOT,
    )
    grid.insert(0, "condition", cond["name"])
    grids[cond["name"]] = grid

all_grids = pd.concat(grids.values(), ignore_index=True)
all_grids.to_csv(f"{TAB_DIR}/multimer_threshold_sweep.csv", index=False)
print(f"\nWrote {TAB_DIR}/multimer_threshold_sweep.csv ({len(all_grids)} cells)")


# %%
# Heatmap helper: one metric as a num_specs x cor_threshold image, points marked.
def _pivot(grid: pd.DataFrame, value: str) -> pd.DataFrame:
    return grid.pivot(index="num_specs", columns="cor_threshold", values=value)


def _draw_heatmap(ax, pt: pd.DataFrame, title: str, cmap: str, vmin, vmax, cbar_label):
    im = ax.imshow(
        pt.to_numpy(), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xticks(range(len(pt.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pt.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pt.index)))
    ax.set_yticklabels(pt.index, fontsize=7)
    ax.set_xlabel("cor_threshold")
    ax.set_ylabel("num_specs")
    ax.set_title(title, fontsize=9)

    # Mark operating points at the nearest grid cell.
    for ns, cor, label, color in [
        (*DB_POINT, "#000000"),
        (*ALT_POINT, "#ffffff"),
    ]:
        if ns < pt.index.min() or ns > pt.index.max():
            continue
        yi = int(np.argmin(np.abs(pt.index.to_numpy() - ns)))
        xi = int(np.argmin(np.abs(pt.columns.to_numpy() - cor)))
        ax.add_patch(
            Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor=color, lw=1.8)
        )
        ax.text(
            xi, yi, label.split()[0], ha="center", va="center", fontsize=6, color=color
        )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=8)


# %%
# One figure per condition: member_frac | stability | jaspar_sim.
for cond in CONDITIONS:
    grid = grids[cond["name"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _draw_heatmap(
        axes[0],
        _pivot(grid, "member_frac"),
        "Primary-cluster fraction\n(reference-blind)",
        "viridis",
        0,
        1,
        "primary size / num_specs",
    )
    _draw_heatmap(
        axes[1],
        _pivot(grid, "stability"),
        "Primary-mode bootstrap stability\n(reference-blind)",
        "magma",
        0,
        1,
        "mean bootstrap consensus cor",
    )
    _draw_heatmap(
        axes[2],
        _pivot(grid, "jaspar_sim"),
        "JASPAR similarity of primary consensus\n(independent readout)",
        "cividis",
        None,
        None,
        "HOMER similarity",
    )
    fig.suptitle(cond["title"], fontsize=12)
    fig.tight_layout()
    out = f"{FIG_DIR}/multimer_sweep_{cond['name']}.svg"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Wrote {out}")
    plt.show()


# %%
# Summary table: metrics at the two marked operating points per condition.
def _nearest_cell(grid: pd.DataFrame, ns: int, cor: float) -> pd.Series:
    sub = grid.copy()
    sub["d"] = (sub["num_specs"] - ns).abs() + 10 * (sub["cor_threshold"] - cor).abs()
    return sub.sort_values("d").iloc[0]


summary_rows = []
for cond in CONDITIONS:
    grid = grids[cond["name"]]
    for ns, cor, label in [DB_POINT, ALT_POINT]:
        cell = _nearest_cell(grid, ns, cor)
        summary_rows.append(
            {
                "condition": cond["name"],
                "point": label,
                "num_specs": int(cell["num_specs"]),
                "cor_threshold": cell["cor_threshold"],
                "member_frac": round(cell["member_frac"], 3),
                "stability": cell["stability"],
                "jaspar_sim": cell["jaspar_sim"],
                "jaspar_p": cell["jaspar_p"],
                "recovery": cell["recovery"],
            }
        )
summary = pd.DataFrame(summary_rows)
print("\n=== Operating-point comparison ===")
print(summary.to_string(index=False))
summary.to_csv(f"{TAB_DIR}/multimer_threshold_sweep_points.csv", index=False)

# %%
