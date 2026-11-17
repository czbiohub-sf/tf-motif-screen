"""Stage 03: compute celltype x timepoint GRNs from a base GRN + scRNA, for ANY arm.

Uses the full multiome RNA counts as input. The same code runs
the CisBP, ZebraMotif, and hybrid arms — only `--arm` changes which base GRN is loaded. Everything
else (RNA, timepoint subset, HVG, annotation, alpha, edge filter) is identical, so a GRN
difference between arms is attributable solely to the motif DB.

Usage:
    python 03_compute_celltype_grns.py --index 3 --arm zebramotif
    python 03_compute_celltype_grns.py --index 3 --arm cisbp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import celloracle as co
import pandas as pd
import scanpy as sc
from matplotlib.colors import to_hex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grnswap import config as C  # noqa: E402


def main() -> None:
    """Build per-celltype GRNs for one timepoint/arm: load RNA + base GRN, run Oracle, save links."""
    ap = argparse.ArgumentParser(
        description="Compute celltype x timepoint GRNs (cisbp | zebramotif | hybrid arm)"
    )
    ap.add_argument("--config", default=None)
    C.add_annotation_argument(ap)
    ap.add_argument(
        "--index", type=int, required=True, help="timepoint index in grn_swap.yaml"
    )
    ap.add_argument("--arm", choices=["cisbp", "zebramotif", "hybrid"], required=True)
    args = ap.parse_args()

    cfg = C.load_config(args.config, annotation=args.annotation)
    tp = C.timepoint(cfg, args.index)
    g = cfg["grn"]
    annotation = g["annotation"]
    dim_reduce = g["dim_reduce"]

    base_grn_path = C.base_grn_for_arm(cfg, tp, args.arm)
    print(
        f"=== Stage 03 | arm={args.arm} | {tp['stage']} / {tp['id']} | base GRN={base_grn_path} ==="
    )

    # RNA: load, subset to timepoint, raw counts, HVG
    layer = cfg["inputs"]["rna_layer"]
    print(
        f"RNA input: {cfg['inputs']['rna_h5ad']} | layer={layer} "
        f"| annotation={annotation}"
    )
    adata = sc.read_h5ad(cfg["inputs"]["rna_h5ad"])
    if layer not in adata.layers:
        raise ValueError(f"RNA input is missing the configured count layer {layer!r}")
    if annotation not in adata.obs:
        raise ValueError(f"RNA input is missing annotation column {annotation!r}")
    if dim_reduce not in adata.obsm:
        raise ValueError(f"RNA input is missing embedding {dim_reduce!r}")
    if g.get("subset_by_timepoint", True):
        col, lab = g["timepoint_col"], tp["rna_timepoint"]
        adata = adata[adata.obs[col].astype(str) == lab].copy()
        if adata.n_obs == 0:
            raise ValueError(f"No RNA cells match {col}=={lab!r}")
        print(f"subset RNA to {col}=={lab!r}: {adata.shape}")
    adata.X = adata.layers[layer].copy()
    if adata.n_vars > g["hvg_n_top_genes"]:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=g["hvg_n_top_genes"], flavor="seurat_v3"
        )
        adata = adata[:, adata.var.highly_variable].copy()
    print(f"RNA for GRN: {adata.shape} | celltypes: {adata.obs[annotation].nunique()}")

    # Base GRN
    base_grn = pd.read_parquet(base_grn_path)
    print(f"base GRN df: {base_grn.shape}")

    # Oracle + KNN imputation
    # CellOracle expects hex strings; some H5AD palettes contain RGB arrays.
    color_key = f"{annotation}_colors"
    if color_key in adata.uns:
        adata.uns[color_key] = [to_hex(color) for color in adata.uns[color_key]]
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=adata, cluster_column_name=annotation, embedding_name=dim_reduce
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)
    oracle.perform_PCA()
    n_cell = oracle.adata.shape[0]
    k = max(1, int(0.025 * n_cell))
    oracle.knn_imputation(
        n_pca_dims=50, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=8
    )
    oracle.to_hdf5(C.oracle_out(cfg, tp, args.arm))

    # Per-celltype GRNs
    t0 = time.time()
    links = oracle.get_links(
        cluster_name_for_GRN_unit=annotation,
        alpha=g["alpha"],
        verbose_level=1,
        test_mode=False,
        n_jobs=2,
    )
    links.filter_links(
        p=g["filter_p"],
        weight=g["filter_weight"],
        threshold_number=g["filter_threshold_number"],
    )
    links.get_network_score()
    links.to_hdf5(file_path=C.links_out(cfg, tp, args.arm))

    csv_dir = C.grn_csv_dir(cfg, tp, args.arm)
    for cluster in links.links_dict.keys():
        links.links_dict[cluster].to_csv(f"{csv_dir}/raw_GRN_for_{cluster}.csv")
    print(
        f"GRN inference: {(time.time()-t0)/60:.1f} min | {len(links.links_dict)} celltypes "
        f"| links -> {C.links_out(cfg, tp, args.arm)} | csvs -> {csv_dir}"
    )


if __name__ == "__main__":
    main()
