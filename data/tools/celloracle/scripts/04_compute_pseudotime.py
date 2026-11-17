"""Stage 04 (KO prep): compute DPT pseudotime on a whole-embryo Oracle for the mesoderm +
neuroectoderm lineages, so the in-silico KO (stage 05) can build a developmental-flow gradient.

Uses the multiome global_annotation fits. Root = NMP centroid;
lineages are defined by the broad multiome cell-type labels.

Usage: python 04_compute_pseudotime.py --index 3 --arm hybrid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import celloracle as co
import numpy as np
import pandas as pd
import scanpy as sc
from celloracle.applications import Pseudotime_calculator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grnswap import config as C  # noqa: E402

LINEAGES = {
    "Lineage_Meso": ["NMPs", "PSM", "Somites", "Muscle", "Adaxial_Cells"],
    "Lineage_NeuroEcto": ["NMPs", "Neural_Posterior", "Neural_Anterior"],
}
ROOT = "NMPs"


def main() -> None:
    """Compute DPT pseudotime per lineage on a saved Oracle (root = NMP centroid) and re-save it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    C.add_annotation_argument(ap)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--arm", choices=["cisbp", "zebramotif", "hybrid"], required=True)
    args = ap.parse_args()

    cfg = C.load_config(args.config, annotation=args.annotation)
    if cfg["grn"]["annotation"] != "global_annotation":
        ap.error("Pseudotime lineages require --annotation global_annotation")
    tp = C.timepoint(cfg, args.index)
    ann = cfg["grn"]["annotation"]
    dim = cfg["grn"]["dim_reduce"]

    oracle_in = C.oracle_out(cfg, tp, args.arm)
    print(f"=== Stage 04 pseudotime | arm={args.arm} | {tp['stage']}/{tp['id']} ===")
    oracle = co.load_hdf5(oracle_in)
    oracle.embedding = oracle.adata.obsm[dim]
    oracle.embedding_name = dim

    pt = Pseudotime_calculator(oracle_object=oracle)
    # keep only lineage clusters that actually exist in this timepoint
    present = set(pt.cluster_list)
    lin = {k: [c for c in v if c in present] for k, v in LINEAGES.items()}
    lin = {k: v for k, v in lin.items() if ROOT in v and len(v) >= 2}
    print("lineages used:", lin)
    pt.set_lineage(lineage_dictionary=lin)

    # root cell = cell closest to the NMP centroid in the embedding
    adata = oracle.adata
    root_sub = adata[adata.obs[ann] == ROOT]
    coords = pd.DataFrame(
        root_sub.obsm[dim][:, :2], columns=["x", "y"], index=root_sub.obs_names
    )
    centroid = coords[["x", "y"]].mean().values
    d = np.sqrt((coords["x"] - centroid[0]) ** 2 + (coords["y"] - centroid[1]) ** 2)
    root_id = d.idxmin()
    pt.set_root_cells(root_cells={k: root_id for k in lin})

    if "X_diffmap" not in pt.adata.obsm:
        sc.pp.neighbors(pt.adata, n_neighbors=20)
        sc.tl.diffmap(pt.adata)
    pt.get_pseudotime_per_each_lineage()
    oracle.adata.obs = pt.adata.obs

    n_pt = oracle.adata.obs["Pseudotime"].notna().sum()
    out = str(
        Path(C.oracle_out(cfg, tp, args.arm)).with_name(
            f"04_{tp['id']}_pseudotime.celloracle.oracle"
        )
    )
    oracle.to_hdf5(out)
    print(f"Pseudotime for {n_pt}/{oracle.adata.n_obs} cells | saved -> {out}")


if __name__ == "__main__":
    main()
