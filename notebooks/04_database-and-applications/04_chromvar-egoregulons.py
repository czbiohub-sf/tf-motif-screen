"""
Compare a TF's ego-regulon stars and ranked target bars across stages.

Stars and bars use the same top targets, with coefficient scales shared across
stages. Edit HYPERPARAMETERS for interactive cells, or use command-line arguments.
"""

# %%
# Imports and project paths
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = (
    Path(__file__).resolve().parents[2] / "data/tools/celloracle"
    if "__file__" in globals()
    else next(
        parent / "data/tools/celloracle"
        for parent in (Path.cwd(), *Path.cwd().parents)
        if (parent / "data/tools/celloracle/src").is_dir()
    )
)
sys.path.insert(0, str(ROOT / "src"))
from grnswap import config as C  # noqa: E402

FIG_DIR = C.PROJECT_ROOT / "notebooks/04_database-and-applications/figures/celloracle"

# %%
# HYPERPARAMETERS
ANNOTATION = "annotation_ML_coarse"
TFS = ["her7"]
CELLTYPE = "PSM"
STAGES = ["all"]
TOP_K = 12
FORMATS = ("svg",)

# Plot style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
C_ACT, C_REP = "#2a78d6", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"


# %%
# Load and rank significant targets
def load_targets(csv: Path, tf: str, p_cutoff: float) -> pd.DataFrame:
    """Significant candidate targets of `tf` in one celltype x timepoint GRN, ranked by |coef|."""
    df = pd.read_csv(csv)
    sub = df[df["source"] == tf].copy()
    sub = sub[pd.to_numeric(sub["p"], errors="coerce") < p_cutoff]
    for column in ("coef_mean", "coef_abs"):
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub[
        np.isfinite(sub["coef_mean"])
        & np.isfinite(sub["coef_abs"])
        & (sub["coef_abs"] > 0)
    ]
    sub = sub.sort_values(["coef_abs", "target"], ascending=[False, True]).reset_index(
        drop=True
    )
    return sub[["target", "coef_mean", "coef_abs", "p", "-logp"]]


# %%
# Draw ego-regulon stars and ranked target bars
def draw_ego(
    ax,
    tf: str,
    sub: pd.DataFrame,
    title: str,
    weight_max: float | None = None,
    top_k: int = 12,
) -> None:
    """Star layout: TF hub at center, top-K targets on a ring; edge width ~ |coef|, color ~ sign."""
    top = sub.head(top_k)
    n = len(top)
    ax.set_axis_off()
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    if n == 0:
        ax.text(
            0,
            0,
            "no significant\ntargets",
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        return

    wmax = weight_max or float(top["coef_abs"].max())
    colors = np.where(top["coef_mean"] > 0, C_ACT, C_REP)
    # ring positions (start at top, go clockwise)
    ang = np.pi / 2 - np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = 1.12
    xs, ys = R * np.cos(ang), R * np.sin(ang)

    for x, y, ca, color in zip(xs, ys, top["coef_abs"], colors):
        lw = 0.6 + 4.4 * (ca / wmax)
        ax.plot(
            [0, x],
            [0, y],
            color=color,
            lw=lw,
            alpha=0.55,
            solid_capstyle="round",
            zorder=1,
        )

    # target nodes + labels
    for x, y, tgt, color in zip(xs, ys, top["target"], colors):
        ax.scatter(
            [x], [y], s=70, color=color, edgecolors="white", linewidths=0.8, zorder=3
        )
        ha = "left" if x > 0.05 else ("right" if x < -0.05 else "center")
        lx = x * 1.24
        ly = y * 1.24
        ax.text(lx, ly, tgt, ha=ha, va="center", fontsize=8, color=INK, zorder=4)

    # hub
    ax.scatter([0], [0], s=520, color="white", edgecolors=INK, linewidths=1.4, zorder=5)
    ax.text(
        0,
        0,
        tf,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        zorder=6,
    )


def draw_bar(
    ax, sub: pd.DataFrame, weight_max: float | None = None, top_k: int = 12
) -> None:
    """Ranked horizontal bar of the top-K targets by |coef|, colored by sign."""
    top = sub.head(top_k).iloc[::-1]  # highest at top
    if len(top) == 0:
        ax.set_axis_off()
        return
    y = np.arange(len(top))
    colors = np.where(top["coef_mean"] > 0, C_ACT, C_REP)
    ax.barh(y, top["coef_abs"], color=colors, height=0.66, zorder=2)
    if weight_max:
        ax.set_xlim(0, weight_max * 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(top["target"], fontsize=8, color=INK)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("coefficient magnitude  |coef|", fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="x", color="#e1e0d9", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.01)


# %%
# Assemble the stage comparison and save its target table
def build_figure(
    tf, all_tables, output, *, annotation, p_cutoff, top_k, formats, cell_counts
):
    """Plot the ego-regulon + ranked-bar panels for one TF across its contexts."""
    weight_max = max(
        (float(sub.coef_abs.max()) for _, _, _, sub in all_tables if len(sub)),
        default=0.0,
    )
    ncol = min(len(all_tables), 3)
    nrow = (len(all_tables) + ncol - 1) // ncol
    figw = max(4.7 * ncol, 7.0)
    fig = plt.figure(figsize=(figw, 7.5 * nrow + 1.0))
    gs = fig.add_gridspec(
        2 * nrow,
        ncol,
        height_ratios=[1.05, 1.0] * nrow,
        hspace=0.5,
        wspace=0.42,
        top=1 - 1.25 / fig.get_figheight(),
        bottom=0.6 / fig.get_figheight(),
        left=0.06,
        right=0.97,
    )

    for j, (title, celltype, stage, sub) in enumerate(all_tables):
        row, col = divmod(j, ncol)
        count = cell_counts.get((celltype, stage))
        panel_title = title + (f" · {count:,} cells" if count is not None else "")
        ax_net = fig.add_subplot(gs[2 * row, col])
        draw_ego(ax_net, tf, sub, panel_title, weight_max, top_k)
        ax_bar = fig.add_subplot(gs[2 * row + 1, col])
        draw_bar(ax_bar, sub, weight_max, top_k)
        ax_bar.text(
            0.98,
            -0.23,
            f"showing {min(top_k, len(sub))} of {len(sub)} significant targets",
            transform=ax_bar.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=MUTED,
        )

    fig.text(
        0.06,
        1 - 0.3 / fig.get_figheight(),
        f"{tf}  ego-regulon",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        1 - 0.6 / fig.get_figheight(),
        f"hybrid ZebraMotif+CisBP · {annotation} · p < {p_cutoff} · top {top_k} per stage",
        ha="left",
        va="center",
        fontsize=9,
        color=INK2,
    )
    fig.text(
        0.06,
        1 - 0.87 / fig.get_figheight(),
        "Blue: positive coefficient · Red: negative coefficient · Width/bar scales shared across stages",
        ha="left",
        va="center",
        fontsize=8,
        color=INK2,
    )
    for fmt in formats:
        fig.savefig(
            output / f"egoregulon_{tf}.{fmt}",
            format=fmt,
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
        )
    plt.close(fig)

    rows = []
    for title, celltype, stage, sub in all_tables:
        rank = np.arange(1, len(sub) + 1)
        table = sub.assign(
            context=title,
            celltype=celltype,
            stage=stage,
            regulator=tf,
            rank=rank,
            annotation=annotation,
            n_cells=cell_counts.get((celltype, stage)),
            selected_top_k=rank <= top_k,
        )
        rows.append(table)
    pd.concat(rows, ignore_index=True).to_csv(
        output / f"ranked_targets_{tf}.csv", index=False
    )
    return output / f"egoregulon_{tf}.{formats[0]}"


# %%
# Load the requested TFs, cell type, and stages
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    C.add_annotation_argument(parser)
    parser.add_argument("--tf", nargs="+", required=True)
    parser.add_argument("--celltype", required=True)
    parser.add_argument(
        "--stage", nargs="+", required=True, help="Configured stages, or all"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--formats", nargs="+", choices=("svg", "png", "pdf"), default=list(FORMATS)
    )
    parser.add_argument(
        "--cell-counts", type=Path, help="TSV: annotation, cell_type, stage, n_cells"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the notebook's figure directory"
    )
    args = parser.parse_args(argv)
    cfg = C.load_config(args.config, annotation=args.annotation, resolve_external=False)
    stages = {row["stage"]: row["id"] for row in cfg["timepoints"]}
    requested = (
        list(stages) if args.stage == ["all"] else list(dict.fromkeys(args.stage))
    )
    if set(requested) - stages.keys() or args.top_k < 1:
        parser.error("Use configured stages (or all) and a positive top-k")
    annotation = cfg["grn"].get("annotation", "global_annotation")
    root = Path(cfg["outputs"]["root"])
    output = args.output_dir or FIG_DIR / annotation / "chromvar_egoregulons"
    counts = {}
    if args.cell_counts:
        frame = pd.read_csv(args.cell_counts, sep="\t")
        counts = {
            (row.cell_type, row.stage): int(row.n_cells)
            for row in frame.loc[frame.annotation.eq(annotation)].itertuples()
        }
    for tf in dict.fromkeys(args.tf):
        tables = [
            (
                f"{args.celltype} @ {stage.replace('somite', '-somite')}",
                args.celltype,
                stage,
                load_targets(
                    root
                    / f"grn_csv/hybrid/03_{stages[stage]}/raw_GRN_for_{args.celltype}.csv",
                    tf,
                    cfg["grn"]["filter_p"],
                ),
            )
            for stage in requested
        ]
        output.mkdir(parents=True, exist_ok=True)
        print(
            build_figure(
                tf,
                tables,
                output,
                annotation=annotation,
                p_cutoff=cfg["grn"]["filter_p"],
                top_k=args.top_k,
                formats=list(dict.fromkeys(args.formats)),
                cell_counts=counts,
            )
        )


# %%
# Run the parameter block in a notebook, or read arguments in the terminal
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main(
            [
                "--annotation",
                ANNOTATION,
                "--tf",
                *TFS,
                "--celltype",
                CELLTYPE,
                "--stage",
                *STAGES,
            ]
        )
    else:
        main()
