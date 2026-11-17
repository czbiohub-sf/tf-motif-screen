#!/usr/bin/env python3
"""
Compare independently selected CisBP and hybrid GRNs on a shared layout.

Each arm nominates its own top targets per TF and retains significant edges
among those genes. Blue highlights additions to the selected network with hybrid.
"""

# %%
# Imports and project paths
import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from grnswap import focused_grn, networkx_layout  # noqa: E402
from grnswap import networkx_rendering as plotting  # noqa: E402

FIG_DIR = C.PROJECT_ROOT / "notebooks/04_database-and-applications/figures/celloracle"
TF_TABLE = C.PROJECT_ROOT / "data/sequences/animaltfdb/tfdb4_tfs.csv"


# %%
# HYPERPARAMETERS
ANNOTATION = "annotation_ML_coarse"
CELLTYPE = "PSM"
STAGES = ["15somite"]
SEEDS = ["tbx6", "tbx16", "her1", "her7"]
TOP_K = 8
BACKGROUND = "faint"
FORMATS = ("png", "svg", "pdf")
LAYOUT_SEED = 17
SPRING_SPACING = 1.5
DPI = 180


# %%
# Select, lay out, and save the requested comparisons
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=C.CONFIG_PATH)
    C.add_annotation_argument(parser)
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--stage", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--background", choices=("none", "faint"), default=BACKGROUND)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=list(FORMATS),
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the notebook's figure directory"
    )
    parser.add_argument("--grn-root", type=Path)
    parser.add_argument(
        "--tf-table",
        type=Path,
        default=TF_TABLE,
        help="TF catalogue CSV with a TF_Name column; defaults to AnimalTFDB4",
    )
    parser.add_argument("--p-cutoff", type=float)
    parser.add_argument("--layout-seed", type=int, default=LAYOUT_SEED)
    parser.add_argument("--spring-spacing", type=float, default=SPRING_SPACING)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args(argv)
    if (
        args.top_k < 1
        or args.dpi < 1
        or not math.isfinite(args.spring_spacing)
        or args.spring_spacing <= 0
    ):
        parser.error("top-k, dpi and spring-spacing must be positive")
    config = C.load_config(
        args.config, annotation=args.annotation, resolve_external=False
    )
    stages = {row["stage"]: row["id"] for row in config["timepoints"]}
    if set(args.stage) - stages.keys():
        parser.error("Unknown stage; choose from " + ", ".join(stages))
    root = Path(config["outputs"]["root"])
    annotation = config["grn"].get("annotation", "global_annotation")
    output = args.output_dir or FIG_DIR / annotation / "grn_networkx"
    grns = args.grn_root or root / "grn_csv"
    seeds = list(dict.fromkeys(args.seeds))
    tf_names = pd.read_csv(
        args.tf_table, usecols=["TF_Name"], dtype=str, keep_default_na=False
    )["TF_Name"].str.strip()
    cutoff = args.p_cutoff if args.p_cutoff is not None else config["grn"]["filter_p"]
    style = {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "svg.hashsalt": "grn-comparison",
        "axes.unicode_minus": False,
    }
    with plt.rc_context(style):
        for stage in dict.fromkeys(args.stage):
            comparison = focused_grn.load_context(
                grns, args.celltype, stages[stage], cutoff
            )
            nodes, edges = focused_grn.select_top_target_subnetwork(
                comparison, seeds, args.top_k
            )
            nodes["is_tf"] = nodes.is_seed | nodes.gene.isin(tf_names)
            networkx_layout.assign_coordinates(
                nodes,
                edges,
                seed=args.layout_seed,
                spring_spacing=args.spring_spacing,
            )
            destination = output / stage / f"top{args.top_k}"
            if set(seeds) != set(SEEDS):
                destination /= "seeds_" + "_".join(
                    re.sub(r"[^a-zA-Z0-9_.-]+", "-", gene) for gene in sorted(seeds)
                )
            artifacts = plotting.render_comparison(
                nodes,
                edges,
                destination,
                celltype=args.celltype,
                stage=stage,
                top_k=args.top_k,
                p_cutoff=cutoff,
                background=args.background,
                formats=dict.fromkeys(args.formats),
                dpi=args.dpi,
            )
            print(next(iter(artifacts.values())), flush=True)
    return 0


# %%
# Run the parameter block in a notebook, or read arguments in the terminal
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main(
            [
                "--annotation",
                ANNOTATION,
                "--celltype",
                CELLTYPE,
                "--stage",
                *STAGES,
                "--seeds",
                *SEEDS,
            ]
        )
    else:
        raise SystemExit(main())
