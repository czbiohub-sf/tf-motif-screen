"""Load matched raw GRNs and select independent top-target neighborhoods."""

from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("cisbp", "hybrid")
METRICS = ("coef_mean", "coef_abs", "p")
COMPARISON_COLUMNS = (
    ["source", "target"]
    + [f"{metric}_{arm}" for metric in METRICS for arm in ARMS]
    + ["present_cisbp", "present_hybrid", "edge_class"]
)


def _read_arm(path: Path) -> pd.DataFrame:
    required = ["source", "target", *METRICS]
    frame = pd.read_csv(
        path, keep_default_na=False, dtype={"source": str, "target": str}
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    frame = frame[required].copy()
    for column in ("source", "target"):
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{path}: empty {column} gene identifier")
    for column in METRICS:
        values = pd.to_numeric(frame[column], errors="coerce")
        missing = (
            frame[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["", "nan", "na", "n/a", "null", "none"])
        )
        invalid = (values.isna() & ~missing) | (values.notna() & ~np.isfinite(values))
        if invalid.any():
            raise ValueError(
                f"{path}: {column} contains malformed or non-finite numeric values"
            )
        frame[column] = values
    if (frame["p"].notna() & ~frame["p"].between(0, 1)).any():
        raise ValueError(f"{path}: p values must lie between 0 and 1")
    if frame["coef_abs"].lt(0).any():
        raise ValueError(f"{path}: coef_abs must be nonnegative")
    # Identical repeated export rows are harmless; conflicting fitted weights
    # must not be silently averaged or chosen by file order.
    frame = frame.drop_duplicates()
    if frame.duplicated(["source", "target"]).any():
        raise ValueError(f"{path}: conflicting duplicate source/target edges")
    return frame


def load_context(
    grn_root: Path, celltype: str, data_id: str, p_cutoff: float = 0.001
) -> pd.DataFrame:
    """Load a matched CisBP/hybrid context, retaining the union of passing edges.

    ``grn_root`` contains ``{cisbp,hybrid}/03_{data_id}/raw_GRN_for_{celltype}.csv``.
    Both files are required, even when one arm has no passing edges. Columns in
    the nonpassing arm preserve its raw values, while truly absent edges are NaN.
    Exact duplicate rows are collapsed; conflicting duplicates and invalid
    measurements raise ``ValueError``. Missing raw numeric values are retained
    as NaN and fail their corresponding gate. The p threshold is strict.
    """
    if (
        isinstance(p_cutoff, bool)
        or not isinstance(p_cutoff, Real)
        or not np.isfinite(p_cutoff)
        or not 0 < p_cutoff <= 1
    ):
        raise ValueError("p_cutoff must be finite and in (0, 1]")
    if not str(celltype).strip() or not str(data_id).strip():
        raise ValueError("celltype and data_id must be nonempty")
    paths = {
        arm: Path(grn_root) / arm / f"03_{data_id}" / f"raw_GRN_for_{celltype}.csv"
        for arm in ARMS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw GRN file(s): " + "; ".join(missing))
    frames = {arm: _read_arm(path) for arm, path in paths.items()}
    renamed = [
        frames[arm].rename(columns={metric: f"{metric}_{arm}" for metric in METRICS})
        for arm in ARMS
    ]
    comparison = renamed[0].merge(
        renamed[1], on=["source", "target"], how="outer", validate="one_to_one"
    )
    for arm in ARMS:
        comparison[f"present_{arm}"] = (
            comparison[f"p_{arm}"].lt(p_cutoff)
            & comparison[f"coef_abs_{arm}"].gt(0)
            & comparison[f"coef_mean_{arm}"].notna()
        )
    comparison = comparison.loc[
        comparison["present_cisbp"] | comparison["present_hybrid"]
    ].copy()
    comparison["edge_class"] = np.select(
        [
            comparison["present_cisbp"] & comparison["present_hybrid"],
            comparison["present_cisbp"],
        ],
        ["shared", "cisbp_only"],
        default="hybrid_only",
    )
    comparison = (
        comparison[COMPARISON_COLUMNS]
        .sort_values(["source", "target"], kind="mergesort")
        .reset_index(drop=True)
    )
    return comparison


def select_top_target_subnetwork(comparison, seeds, top_k=8):
    """Select each arm's top-K targets and mark its induced significant edges.

    The union is retained for one shared layout. Full-GRN presence stays separate
    from membership in either selected network.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    seeds = sorted(
        {str(gene).strip() for gene in ([seeds] if isinstance(seeds, str) else seeds)}
    )
    if not seeds or "" in seeds:
        raise ValueError("At least one nonempty seed gene is required")
    selected, nominated = {}, {}
    for arm in ARMS:
        passing = comparison.loc[
            comparison[f"present_{arm}"] & comparison.source.isin(seeds)
        ]
        weights = pd.to_numeric(passing[f"coef_abs_{arm}"], errors="coerce")
        if (~np.isfinite(weights) | weights.le(0)).any():
            raise ValueError(f"Passing {arm} edges require positive finite weights")
        top = (
            passing.sort_values(
                ["source", f"coef_abs_{arm}", "target"], ascending=[True, False, True]
            )
            .groupby("source", sort=False)
            .head(top_k)
        )
        selected[arm] = set(seeds) | set(top.target)
        nominated[arm] = set(zip(top.source, top.target))
    genes = selected["cisbp"] | selected["hybrid"]
    nodes = pd.DataFrame(
        {"gene": sorted(genes, key=lambda gene: (gene not in seeds, gene))}
    )
    nodes["is_seed"] = nodes.gene.isin(seeds)
    for arm in ARMS:
        nodes[f"selected_{arm}"] = nodes.gene.isin(selected[arm])
    edges = comparison.loc[
        comparison.source.isin(genes) & comparison.target.isin(genes)
    ].copy()
    for arm in ARMS:
        edges[f"nominated_{arm}"] = [
            (source, target) in nominated[arm]
            for source, target in zip(edges.source, edges.target)
        ]
        edges[f"in_selected_{arm}"] = (
            edges[f"present_{arm}"]
            & edges.source.isin(selected[arm])
            & edges.target.isin(selected[arm])
        )
    edges["selection_edge_class"] = np.select(
        [
            edges.in_selected_cisbp & edges.in_selected_hybrid,
            edges.in_selected_cisbp,
            edges.in_selected_hybrid,
        ],
        ["shared", "cisbp_only", "hybrid_only"],
        default="not_selected",
    )
    return nodes, edges.sort_values(["source", "target"]).reset_index(drop=True)
