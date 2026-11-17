"""Compare each arm's selected GRN using shared node and label coordinates."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from .focused_grn import ARMS

CISBP_COLOR = "#efb64d"
HYBRID_COLOR = "#2a78d6"
TF_COLOR = "#8052a5"
GENE_COLOR = "#b5b5b5"


def draw_panel(ax, nodes, edges, *, arm, scale, background):
    """Highlight edges gained in the hybrid selection over the CisBP selection."""
    positions = nodes.set_index("gene")[["x", "y"]].T.to_dict("list")
    genes = nodes.gene.tolist()
    sizes = np.where(nodes.is_seed, 150, 32)
    graph = nx.DiGraph()
    graph.add_nodes_from(genes)
    graph.add_edges_from(zip(edges.source, edges.target))
    active = set(nodes.loc[nodes[f"selected_{arm}"], "gene"])
    foreground = edges.source.isin(nodes.loc[nodes.is_seed, "gene"])
    # Background edges go first; the faint CisBP silhouette is only a guide to
    # the hybrid panel, and is never exported as fitted CisBP edges.
    for is_foreground in (False, True):
        pairs, colors, widths = [], [], []
        for row in edges.loc[foreground.eq(is_foreground)].itertuples(index=False):
            present = getattr(row, f"in_selected_{arm}")
            ghost = arm == "cisbp" and not present and row.in_selected_hybrid
            if not present and not ghost:
                continue
            if background == "none" and (not is_foreground or ghost):
                continue
            color = CISBP_COLOR if row.in_selected_cisbp else HYBRID_COLOR
            alpha = 0.045 if ghost else 0.8 if is_foreground else 0.07
            weight_arm = "hybrid" if ghost else arm
            pairs.append((row.source, row.target))
            colors.append(to_rgba(color, alpha))
            widths.append(
                0.6 + 1.4 * np.sqrt(getattr(row, f"coef_abs_{weight_arm}") / scale)
                if is_foreground
                else 0.45
            )
        if pairs:
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=ax,
                nodelist=genes,
                node_size=sizes,
                edgelist=pairs,
                edge_color=colors,
                width=widths,
                arrows=True,
                arrowstyle="-|>" if is_foreground else "-",
                arrowsize=9,
                connectionstyle="arc3,rad=0.035",
            )
    for row, size in zip(nodes.itertuples(index=False), sizes):
        alpha = 1.0 if row.gene in active else 0.16
        ax.scatter(
            row.x,
            row.y,
            s=size,
            color=TF_COLOR if row.is_tf else GENE_COLOR,
            edgecolors="#233440" if row.is_seed else "white",
            linewidths=1.4 if row.is_seed else 0.6,
            alpha=alpha,
            zorder=3,
        )
        ax.text(
            row.label_x,
            row.label_y,
            row.gene,
            ha="center",
            va="center",
            fontsize=9 if row.is_seed else 8,
            color="#233440",
            alpha=alpha,
            zorder=4,
        )
    xy = nodes[["x", "y", "label_x", "label_y"]].to_numpy().reshape(-1, 2)
    low, high = xy.min(axis=0), xy.max(axis=0)
    margin = max(float((high - low).max()) * 0.12, 0.2)
    ax.set(
        xlim=(low[0] - margin, high[0] + margin),
        ylim=(low[1] - margin, high[1] + margin),
        aspect="equal",
        title="CisBP only" if arm == "cisbp" else "CisBP + ZebraMotif",
    )
    ax.set_axis_off()


def render_comparison(
    nodes,
    edges,
    destination,
    *,
    celltype,
    stage,
    top_k,
    p_cutoff,
    background="faint",
    formats=("png", "svg", "pdf"),
    dpi=180,
):
    """Save the paired figure and the selected nodes and fitted edge tables."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scale = max(
        (
            float(edges.loc[edges[f"in_selected_{arm}"], f"coef_abs_{arm}"].max())
            for arm in ARMS
            if edges[f"in_selected_{arm}"].any()
        ),
        default=1.0,
    )
    fig, axes = plt.subplots(2, 1, figsize=(8, 14), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.09, hspace=0.15)
    label = (
        stage.removesuffix("somite") + " somites" if stage.endswith("somite") else stage
    )
    fig.suptitle(
        f"{celltype.replace('_', ' ')} · {label} · top {top_k} per seed in each arm · p < {p_cutoff:g}",
        fontsize=12,
    )
    for ax, arm in zip(axes, ARMS):
        draw_panel(ax, nodes, edges, arm=arm, scale=scale, background=background)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=TF_COLOR,
            markeredgecolor="#233440",
            markersize=10,
            label="Focal TF",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=TF_COLOR,
            markersize=5,
            label="Other TF",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=GENE_COLOR,
            markersize=5,
            label="Target gene",
        ),
        Line2D([], [], color=CISBP_COLOR, label="In CisBP selection"),
        Line2D([], [], color=HYBRID_COLOR, label="Added in hybrid selection"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    stem = f"{celltype}__{stage}__top{top_k}__{background}"
    artifacts = {}
    try:
        for fmt in formats:
            path = destination / f"{stem}.{fmt}"
            metadata = (
                {"Date": None}
                if fmt == "svg"
                else {"CreationDate": None, "ModDate": None}
                if fmt == "pdf"
                else None
            )
            fig.savefig(path, dpi=dpi, facecolor="white", metadata=metadata)
            artifacts[fmt] = path
    finally:
        plt.close(fig)
    nodes.to_csv(destination / f"{stem}.nodes.csv", index=False, float_format="%.10g")
    tables = []
    for arm in ARMS:
        passing = edges.loc[edges[f"in_selected_{arm}"]]
        table = passing[
            ["source", "target", "edge_class", "selection_edge_class"]
        ].copy()
        table.insert(0, "arm", arm)
        for metric in ("coef_mean", "coef_abs", "p"):
            table[metric] = passing[f"{metric}_{arm}"]
        table["added"] = (arm == "hybrid") & ~passing.in_selected_cisbp
        table["nominated"] = passing[f"nominated_{arm}"]
        table["displayed"] = passing.source.isin(nodes.loc[nodes.is_seed, "gene"]) | (
            background == "faint"
        )
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(
        destination / f"{stem}.edges.csv", index=False, float_format="%.10g"
    )
    return artifacts
