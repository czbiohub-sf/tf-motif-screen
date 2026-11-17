"""
Plot Zebrahub-Multiome cells and the SEACells used by chromVAR.

Metacell markers show mean member-cell coordinates on the saved joint RNA/ATAC
(WNN) UMAP. The >=20-cell cutoff matches data/tools/chromvar/02_deviations.R.
RNA-only panels use the saved RNA UMAP, H5AD annotations, and saved colors.
Only metadata and embeddings are read from the H5AD; counts stay on disk.
"""

# %%
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from af3tfm.config import PATHS
from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)

H5AD_PATH = PATHS["zebrahub_multiome_rna_h5ad"]
ASSIGNMENTS_PATH = DATA_DIR / "tools/chromvar/results/seacells_assignment.csv"
EMBEDDING = "X_wnn.umap"
RNA_EMBEDDING = "X_umap.rna"

# HYPERPARAMETERS
MIN_CELLS = 20
HIGHLIGHTS = []  # e.g. ["TDR118_SEACell-85", "TDR118_SEACell-45"]
RNA_ONLY = False
COMPARE_ANNOTATIONS = False  # Compare annotation_ML_coarse and global_annotation.
CELLTYPE_COLUMN = "annotation_ML_coarse"  # Also annotation_ML or global_annotation.


# %%
# Load cell metadata and metacell assignments
def load_umap(
    h5ad_path: Path, embedding: str, celltype_column: str | None = None
) -> pd.DataFrame:
    """Read cell identities and a saved UMAP without loading expression counts."""
    with h5py.File(h5ad_path, "r") as handle:
        if embedding not in handle["obsm"]:
            available = ", ".join(handle["obsm"])
            raise ValueError(f"Embedding {embedding!r} missing; available: {available}")
        obs = ad.io.read_elem(handle["obs"])
        coords = np.asarray(ad.io.read_elem(handle["obsm"][embedding]))
    if celltype_column is not None and celltype_column not in obs:
        raise ValueError(f"Cell-type column {celltype_column!r} missing from H5AD obs.")
    if not obs.index.is_unique:
        raise ValueError(
            "H5AD cell barcodes must be unique, including dataset suffixes."
        )
    if coords.shape != (len(obs), 2) or not np.isfinite(coords).all():
        raise ValueError(
            "The selected UMAP must contain two finite coordinates per cell."
        )
    cells = pd.DataFrame(coords, index=obs.index, columns=["umap_1", "umap_2"])
    if celltype_column is not None:
        cells["celltype"] = obs[celltype_column].astype("string").fillna("Unassigned")
    return cells


def attach_metacells(
    cells: pd.DataFrame, assignments_path: Path, min_cells: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join exact integrated barcodes and retain chromVAR's SEACell partition."""
    assignments = pd.read_csv(assignments_path)
    required = {"cell", "metacell", "dataset", "dev_stage", "celltype"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignment CSV is missing columns: {sorted(missing)}")
    if assignments[list(required)].isna().any().any():
        raise ValueError("Assignment CSV contains missing identifiers or annotations.")
    if not assignments["cell"].is_unique:
        raise ValueError("Assignment CSV has duplicate cell barcodes.")

    # Preserve the dataset suffix: bare 10x barcodes repeat across samples.
    cells = cells.join(assignments.set_index("cell"), validate="one_to_one")
    n_matched = cells["metacell"].notna().sum()
    if not n_matched:
        raise ValueError(
            "No matching cells; use the integrated multiome H5AD barcodes."
        )
    print(f"Matched {n_matched:,}/{len(cells):,} UMAP cells to SEACells.")

    # Count on the chromVAR assignment table, before any UMAP subset is applied.
    sizes = assignments["metacell"].value_counts()
    cells["retained"] = cells["metacell"].map(sizes).ge(min_cells)
    members = cells.loc[cells["retained"]]
    if members.empty:
        raise ValueError(f"No plotted metacells meet the >= {min_cells}-cell cutoff.")
    grouped = members.groupby("metacell", sort=True)
    centers = grouped.agg(
        umap_1=("umap_1", "mean"),
        umap_2=("umap_2", "mean"),
        n_cells_on_umap=("umap_1", "size"),
        dataset=("dataset", "first"),
        dev_stage=("dev_stage", "first"),
        celltype=("celltype", lambda x: x.value_counts().idxmax()),
    )
    centers["n_cells"] = sizes.reindex(centers.index)
    centers["celltype_purity"] = grouped["celltype"].agg(
        lambda x: x.value_counts(normalize=True).max()
    )
    print(
        f"Retained {len(centers):,} metacells covering {len(members):,} UMAP cells "
        f"(>= {min_cells} cells/metacell)."
    )
    return cells, centers


# %%
# Shared plotting helpers
def format_umap(ax: plt.Axes) -> None:
    """Give all panels the same unscaled UMAP axes."""
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[], aspect="equal")
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    for extension in ("png", "svg"):
        path = figure_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.show()
    plt.close(fig)


def celltype_colors(labels: pd.Series) -> dict:
    """Keep cell-type colors consistent between the RNA and WNN plots."""
    categories = sorted(labels.unique())
    palette = [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(categories)}
    colors["Unassigned"] = "#bdbdbd"
    return colors


def load_saved_celltype_colors(h5ad_path: Path, column: str) -> dict | None:
    """Read the multiome palette in its original categorical annotation order."""
    with h5py.File(h5ad_path, "r") as handle:
        color_key = f"{column}_colors"
        if color_key not in handle.get("uns", {}):
            return None
        labels = ad.io.read_elem(handle["obs"][column])
        if not isinstance(labels, pd.Categorical):
            return None
        palette = ad.io.read_elem(handle["uns"][color_key])
    return dict(zip(labels.categories, palette, strict=True))


def draw_celltypes(
    ax: plt.Axes, cells: pd.DataFrame, colors: dict | None = None
) -> None:
    """Shuffle drawing order so no cell type systematically covers the others."""
    labels = cells["celltype"]
    colors = colors if colors is not None else celltype_colors(labels)
    order = np.random.default_rng(1).permutation(len(cells))
    ax.scatter(
        cells["umap_1"].iloc[order],
        cells["umap_2"].iloc[order],
        c=labels.map(colors).iloc[order].tolist(),
        s=1,
        linewidths=0,
        rasterized=True,
    )
    format_umap(ax)


def celltype_legend(
    ax: plt.Axes, labels: pd.Series, colors: dict, title: str | None = None
) -> None:
    """Place cell-type labels outside the UMAP panel."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=colors[label],
            label=label.replace("_", " "),
            markersize=5,
        )
        for label in sorted(labels.unique())
    ]
    ax.legend(
        handles=handles,
        title=title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        borderaxespad=0,
    )


# %%
# RNA cell types and annotation levels
def plot_rna_celltypes(
    cells: pd.DataFrame,
    figure_dir: Path,
    title: str = "Zebrahub-Multiome | RNA UMAP",
    name: str = "zebrahub_multiome_rna_umap_celltypes",
    colors: dict | None = None,
) -> None:
    """Plot every cell at its saved RNA UMAP coordinates, colored by cell type."""
    labels = cells["celltype"]
    colors = colors if colors is not None else celltype_colors(labels)
    fig, ax = plt.subplots(figsize=(10, max(8, 0.2 * labels.nunique() + 1)))
    draw_celltypes(ax, cells, colors)
    celltype_legend(ax, labels, colors, title="Cell type")
    ax.set(
        xlabel="RNA UMAP 1",
        ylabel="RNA UMAP 2",
        title=f"{title}\n{len(cells):,} cells | {labels.nunique()} cell types",
    )
    fig.tight_layout()
    save_figure(fig, figure_dir, name)


def plot_annotation_levels(h5ad_path: Path, embedding: str, figure_dir: Path) -> None:
    """Compare both coarser annotations on identical cells and RNA coordinates."""
    columns = ("annotation_ML_coarse", "global_annotation")
    panels = [load_umap(h5ad_path, embedding, column) for column in columns]
    pd.testing.assert_index_equal(panels[0].index, panels[1].index)
    np.testing.assert_array_equal(
        panels[0][["umap_1", "umap_2"]], panels[1][["umap_1", "umap_2"]]
    )
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    palettes = [load_saved_celltype_colors(h5ad_path, column) for column in columns]
    for column, cells, colors, ax in zip(columns, panels, palettes, axes):
        colors = colors if colors is not None else celltype_colors(cells["celltype"])
        draw_celltypes(ax, cells, colors)
        celltype_legend(ax, cells["celltype"], colors, title="Cell type")
        ax.set(
            xlabel="RNA UMAP 1",
            ylabel="RNA UMAP 2",
            title=f"{column}\n{cells['celltype'].nunique()} annotation groups",
        )
    fig.suptitle(
        f"Zebrahub-Multiome | Same {len(panels[0]):,} cells and RNA UMAP coordinates",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, figure_dir, "zebrahub_multiome_rna_umap_annotation_levels")

    # Save the individual panels after displaying the complete comparison.
    for column, cells, colors in zip(columns, panels, palettes):
        plot_rna_celltypes(
            cells,
            figure_dir,
            title=f"Zebrahub-Multiome | RNA UMAP ({column})",
            name=f"zebrahub_multiome_rna_umap_{column}",
            colors=colors,
        )


# %%
# Metacell overview and selected SEACells
def plot_overview(cells: pd.DataFrame, centers: pd.DataFrame, figure_dir: Path) -> None:
    """Compare single-cell annotations with metacell centers in the same UMAP."""
    labels = cells["celltype"].fillna("Unassigned")
    colors = celltype_colors(labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    draw_celltypes(axes[0], cells.assign(celltype=labels), colors)
    axes[0].set_title(f"Zebrahub-Multiome | {len(cells):,} cells")
    axes[1].scatter(
        cells["umap_1"],
        cells["umap_2"],
        s=1,
        c="#dedede",
        linewidths=0,
        rasterized=True,
    )
    axes[1].scatter(
        centers["umap_1"],
        centers["umap_2"],
        c=centers["celltype"].map(colors).tolist(),
        s=18,
        edgecolors="white",
        linewidths=0.35,
    )
    axes[1].set_title(f"{len(centers):,} SEACell metacells | majority cell type")
    celltype_legend(axes[1], labels, colors)
    for ax in axes:
        format_umap(ax)
    fig.text(
        0.5,
        0.01,
        "Metacell markers: mean of member-cell UMAP coordinates.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, figure_dir, "zebrahub_multiome_umap_metacells")


def plot_highlights(
    cells: pd.DataFrame, centers: pd.DataFrame, metacells: list[str], figure_dir: Path
) -> None:
    """Highlight every member cell of the requested, dataset-qualified SEACells."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        cells["umap_1"],
        cells["umap_2"],
        c="#dedede",
        s=1,
        linewidths=0,
        rasterized=True,
    )
    palette = plt.get_cmap("tab10").colors
    for i, metacell in enumerate(metacells):
        color = palette[i % len(palette)]
        members = cells.loc[cells["metacell"].eq(metacell)]
        center = centers.loc[metacell]
        ax.scatter(
            members["umap_1"],
            members["umap_2"],
            c=[color],
            s=9,
            linewidths=0,
            label=f"{metacell} ({len(members)} cells)",
        )
        ax.scatter(
            center["umap_1"],
            center["umap_2"],
            c=[color],
            marker="D",
            s=65,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
    format_umap(ax)
    ax.set_title("SEACell members on the Zebrahub-Multiome UMAP")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Diamonds: mean of member-cell UMAP coordinates.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, figure_dir, "zebrahub_multiome_umap_highlighted_metacells")


# %%
# Read the saved embedding
rna_only = RNA_ONLY or COMPARE_ANNOTATIONS
embedding = RNA_EMBEDDING if rna_only else EMBEDDING
if MIN_CELLS < 1:
    raise ValueError("MIN_CELLS must be positive.")
if rna_only and HIGHLIGHTS:
    raise ValueError("RNA-only plots cannot be combined with metacell highlights.")

print(f"Reading {embedding} from {H5AD_PATH}")
if not COMPARE_ANNOTATIONS:
    cells = load_umap(H5AD_PATH, embedding, CELLTYPE_COLUMN if rna_only else None)

# %%
# Summarize the retained chromVAR metacells
if not rna_only:
    cells, centers = attach_metacells(cells, ASSIGNMENTS_PATH, MIN_CELLS)
    highlights = list(dict.fromkeys(HIGHLIGHTS))
    unknown = set(highlights) - set(centers.index)
    if unknown:
        raise ValueError(
            f"Metacells absent or below the size cutoff: {sorted(unknown)}"
        )
    centers.to_csv(TAB_DIR / "zebrahub_multiome_metacells.tsv", sep="\t")

# %%
# Single cells and metacell centers on the joint RNA/ATAC UMAP
if not rna_only:
    plot_overview(cells, centers, FIG_DIR)

# %%
# Member cells of selected SEACells
if not rna_only and highlights:
    plot_highlights(cells, centers, highlights, FIG_DIR)

# %%
# RNA UMAP colored by the selected annotation
if RNA_ONLY and not COMPARE_ANNOTATIONS:
    colors = load_saved_celltype_colors(H5AD_PATH, CELLTYPE_COLUMN)
    if CELLTYPE_COLUMN == "annotation_ML_coarse":
        plot_rna_celltypes(cells, FIG_DIR, colors=colors)
    else:
        plot_rna_celltypes(
            cells,
            FIG_DIR,
            title=f"Zebrahub-Multiome | RNA UMAP ({CELLTYPE_COLUMN})",
            name=f"zebrahub_multiome_rna_umap_{CELLTYPE_COLUMN}",
            colors=colors,
        )

# %%
# Compare coarse and broad annotations on identical RNA coordinates
if COMPARE_ANNOTATIONS:
    plot_annotation_levels(H5AD_PATH, embedding, FIG_DIR)
