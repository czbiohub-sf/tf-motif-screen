"""Stage 05: in-silico KO simulation + perturbation score (PS), one arm at a time.

For a given multiome arm (cisbp|zebramotif|hybrid) and timepoint:
  1. load the pseudotime Oracle (stage 04) + Links; fit the GRN for simulation.
  2. build the developmental-flow gradient (once) from DPT pseudotime.
  3. for each KO gene: simulate_shift(gene=0) -> transition prob -> embedding shift;
     then per lineage compute the inner product of the KO shift vs the developmental flow
     (negative IP = KO opposes normal development = the TF is a positive driver).

TFs absent from an arm's fitted regulators are recorded as not perturbable.
Output: comparison/ko_perturbation_scores_<arm>.csv (appended per run).

Usage: python 05_ko_simulation.py --index 3 --arm hybrid --genes tbxta,tbx16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import celloracle as co
import numpy as np
import pandas as pd
from celloracle.applications import Gradient_calculator, Oracle_development_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grnswap import config as C  # noqa: E402

LINEAGE_CLUSTERS = {
    "Lineage_Meso": ["NMPs", "PSM", "Somites", "Muscle", "Adaxial_Cells"],
    "Lineage_NeuroEcto": ["NMPs", "Neural_Posterior", "Neural_Anterior"],
}


def main() -> None:
    """Simulate each requested gene KO and score it against developmental flow, per lineage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    C.add_annotation_argument(ap)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--arm", choices=["cisbp", "zebramotif", "hybrid"], required=True)
    ap.add_argument("--genes", required=True, help="comma-separated KO genes")
    args = ap.parse_args()

    cfg = C.load_config(args.config, annotation=args.annotation)
    if cfg["grn"]["annotation"] != "global_annotation":
        ap.error("KO lineages require --annotation global_annotation")
    tp = C.timepoint(cfg, args.index)
    ann = cfg["grn"]["annotation"]
    genes = [g for g in args.genes.split(",") if g]

    pt_oracle = str(
        Path(C.oracle_out(cfg, tp, args.arm)).with_name(
            f"04_{tp['id']}_pseudotime.celloracle.oracle"
        )
    )
    links_path = C.links_out(cfg, tp, args.arm)
    print(
        f"=== Stage 05 KO | arm={args.arm} | {tp['stage']}/{tp['id']} | genes={genes} ==="
    )
    oracle = co.load_hdf5(pt_oracle)
    links = co.load_hdf5(links_path)

    # Fit GRN for simulation
    links.filter_links()
    oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
    oracle.fit_GRN_for_simulation(
        alpha=cfg["grn"]["alpha"], use_cluster_specific_TFdict=True
    )
    regulators = set(getattr(oracle, "active_regulatory_genes", []))
    print(f"perturbable regulators in {args.arm} GRN: {len(regulators)}")

    # Prime delta_embedding (calculate_p_mass needs a prior simulation)
    oracle.adata.obs["Pseudotime"] = (
        oracle.adata.obs["Pseudotime"].astype(float).fillna(0)
    )
    n_grid, n_neighbors = 40, 200
    prime = next((g for g in genes if g in regulators), None) or (
        next(iter(regulators)) if regulators else None
    )
    if prime is None:
        raise RuntimeError(f"no perturbable regulators in {args.arm} GRN")
    oracle.simulate_shift(
        perturb_condition={prime: 0.0}, ignore_warning=True, n_propagation=3
    )
    oracle.estimate_transition_prob(
        n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1
    )
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    # Developmental-flow gradient (computed once)
    oracle.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=n_neighbors)
    sugg = np.linspace(oracle.total_p_mass.min(), oracle.total_p_mass.max() / 2, 12)
    min_mass = math.floor(sugg[5])
    oracle.calculate_mass_filter(min_mass=min_mass, plot=False)

    grad = Gradient_calculator(oracle_object=oracle, pseudotime_key="Pseudotime")
    grad.calculate_p_mass(smooth=0.8, n_grid=n_grid, n_neighbors=n_neighbors)
    grad.calculate_mass_filter(min_mass=min_mass, plot=False)
    try:
        grad.transfer_data_into_grid(
            args={"method": "knn", "n_knn": getattr(oracle, "k_knn_imputation", 50)},
            plot=False,
        )
    except Exception:
        grad.transfer_data_into_grid(
            args={"method": "polynomial", "n_poly": 3}, plot=False
        )
    grad.calculate_gradient()

    lineages = {
        k: [
            c
            for c in v
            if c
            in set(
                oracle.adata.obs[ann].cat.categories
                if hasattr(oracle.adata.obs[ann], "cat")
                else oracle.adata.obs[ann].unique()
            )
        ]
        for k, v in LINEAGE_CLUSTERS.items()
    }

    rows = []
    for gene in genes:
        if gene not in regulators:
            print(
                f"  {gene}: NOT perturbable in {args.arm} GRN (not a regulator) -> KO impossible"
            )
            rows.append(
                {
                    "arm": args.arm,
                    "timepoint": tp["stage"],
                    "gene": gene,
                    "perturbable": False,
                    "lineage": "",
                    "mean_ip": np.nan,
                    "frac_negative_ip": np.nan,
                    "n_grid": 0,
                }
            )
            continue
        try:
            oracle.simulate_shift(
                perturb_condition={gene: 0.0}, ignore_warning=True, n_propagation=3
            )
            oracle.estimate_transition_prob(
                n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1
            )
            oracle.calculate_embedding_shift(sigma_corr=0.05)
        except Exception as e:  # noqa: BLE001
            print(f"  {gene}: simulate failed: {type(e).__name__}: {e}")
            rows.append(
                {
                    "arm": args.arm,
                    "timepoint": tp["stage"],
                    "gene": gene,
                    "perturbable": True,
                    "lineage": "ERROR",
                    "mean_ip": np.nan,
                    "frac_negative_ip": np.nan,
                    "n_grid": 0,
                }
            )
            continue

        for lin_name, clusters in lineages.items():
            cell_idx = np.where(oracle.adata.obs[ann].isin(clusters))[0]
            if len(cell_idx) < 50:
                continue
            dev = Oracle_development_module()
            dev.load_differentiation_reference_data(gradient_object=grad)
            dev.load_perturb_simulation_data(
                oracle_object=oracle, cell_idx_use=cell_idx, name=lin_name
            )
            dev.calculate_inner_product()
            dev.calculate_digitized_ip(n_bins=10)
            ip = np.asarray(dev.inner_product)
            ip = ip[~np.isnan(ip)]
            rows.append(
                {
                    "arm": args.arm,
                    "timepoint": tp["stage"],
                    "gene": gene,
                    "perturbable": True,
                    "lineage": lin_name,
                    "mean_ip": float(np.mean(ip)) if len(ip) else np.nan,
                    "frac_negative_ip": float(np.mean(ip < 0)) if len(ip) else np.nan,
                    "n_grid": int(len(ip)),
                }
            )
            print(
                f"  {gene} [{lin_name}]: mean_IP={np.mean(ip):.4f} frac_neg={np.mean(ip<0):.2f} (n_grid={len(ip)})"
            )

    out = Path(C.comparison_dir(cfg)) / f"ko_perturbation_scores_{args.arm}.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
