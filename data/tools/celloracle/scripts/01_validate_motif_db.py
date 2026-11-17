"""Validate the converted AF3 motif DB against the RNA gene-symbol namespace and CellOracle.

Runs three checks, cheapest first, so the important verdicts print even if the scan is slow:
  1. Symbol overlap: how many ZebraMotif (and CisBP) motif factors match the RNA anndata var_names?
     This is the make-or-break check — CellOracle silently drops factors absent from var_names.
  2. Motif loading: load the .pfm/.motif2factors with gimmemotifs and confirm .factors populate.
  3. Tiny scan: run tfi.scan() on a few hundred peaks to confirm an end-to-end base GRN forms.

Usage:
    python 01_validate_motif_db.py            # uses paths from grn_swap.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import pandas as pd
from celloracle import motif_analysis as ma
from gimmemotifs.motif import read_motifs as _gimme_read_motifs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from grnswap import config as C  # noqa: E402

CFG = C.load_config()


def read_factor_set_from_m2f(path: str) -> set[str]:
    """Return the set of factor names (column 2) from a gimmemotifs .motif2factors.txt."""
    factors: set[str] = set()
    with open(path) as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                factors.add(parts[1])
    return factors


def main() -> None:
    """Run the three validation checks in order and save the overlap tables for the report."""
    zebramotif_prefix = CFG["motif_db"]["zebramotif_prefix"]
    new_pfm = zebramotif_prefix + ".pfm"
    new_m2f = zebramotif_prefix + ".motif2factors.txt"
    old_m2f = CFG["motif_db"]["cisbp_prefix"] + ".motif2factors.txt"
    rna_path = CFG["inputs"]["rna_h5ad"]

    # Symbol overlap
    print("=" * 70)
    print("CHECK 1 — gene-symbol overlap with RNA var_names")
    print("=" * 70)
    adata = ad.read_h5ad(rna_path, backed="r")
    var_names = set(map(str, adata.var_names))
    print(f"RNA anndata: {rna_path}")
    print(f"  n_genes (var_names): {len(var_names)}")
    print(f"  example var_names: {list(adata.var_names[:8])}")

    zm_factors = read_factor_set_from_m2f(new_m2f)
    cisbp_factors = read_factor_set_from_m2f(old_m2f)
    zm_hit = zm_factors & var_names
    cisbp_hit = cisbp_factors & var_names
    print(
        f"\nZebraMotif factors: {len(zm_factors)} unique | match var_names: {len(zm_hit)}"
        f" ({100*len(zm_hit)/max(1,len(zm_factors)):.1f}%)"
    )
    print(
        f"CisBP factors: {len(cisbp_factors)} unique | match var_names: {len(cisbp_hit)}"
        f" ({100*len(cisbp_hit)/max(1,len(cisbp_factors)):.1f}%)"
    )
    rescued = zm_hit - cisbp_hit
    print(
        f"\n>>> TFs expressed in RNA gained ONLY via the ZebraMotif DB (rescued): {len(rescued)}"
    )
    print(
        f">>> TFs present in CisBP, expressed in RNA, NOT in ZebraMotif: {len(cisbp_hit - zm_hit)}"
    )
    print(f"    sample rescued: {sorted(list(rescued))[:25]}")

    # Save the overlap tables for the report
    out = Path(CFG["outputs"]["root"]) / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(sorted(rescued)).to_csv(
        out / "factors_rescued_by_zebramotif.csv", index=False, header=["factor"]
    )
    pd.DataFrame(
        {
            "metric": [
                "zebramotif_factors",
                "zebramotif_in_rna",
                "cisbp_factors",
                "cisbp_in_rna",
                "rescued",
                "lost",
            ],
            "value": [
                len(zm_factors),
                len(zm_hit),
                len(cisbp_factors),
                len(cisbp_hit),
                len(rescued),
                len(cisbp_hit - zm_hit),
            ],
        }
    ).to_csv(out / "symbol_overlap_summary.csv", index=False)
    print(
        f"\nsaved: {out/'symbol_overlap_summary.csv'} , {out/'factors_rescued_by_zebramotif.csv'}"
    )

    # Motif loading
    print("\n" + "=" * 70)
    print("CHECK 2 — load motifs into gimmemotifs and confirm .factors populate")
    print("=" * 70)
    motifs = load_motifs(new_pfm)
    print(f"loaded {len(motifs)} motifs from {new_pfm}")
    with_factors = [m for m in motifs if _motif_factors(m)]
    print(f"motifs with non-empty .factors: {len(with_factors)} / {len(motifs)}")
    if with_factors:
        ex = with_factors[0]
        print(f"  example: id={ex.id}  factors={_motif_factors(ex)[:5]}")
    if len(with_factors) == 0:
        print(
            "!!! WARNING: no factors populated — CellOracle base GRN would be empty. "
            "The .motif2factors.txt sibling was not picked up."
        )

    # Test scan on a few hundred peaks to confirm an end-to-end base GRN forms
    print("\n" + "=" * 70)
    print("CHECK 3 — tiny tfi.scan() end-to-end (a few hundred peaks)")
    print("=" * 70)
    try:
        tiny_scan(motifs)
    except Exception as e:  # noqa: BLE001
        print(f"!!! tiny scan raised: {type(e).__name__}: {e}")
        print("    (symbol-overlap + motif-loading verdicts above are the gating ones)")


def load_motifs(pfm_path: str):
    """Load motifs with factor info, preferring CellOracle's reader over gimmemotifs."""
    if hasattr(ma, "read_motifs"):
        return ma.read_motifs(pfm_path)
    return _gimme_read_motifs(pfm_path)


def _motif_factors(m) -> list[str]:
    """Return a motif's factor names as a flat list, flattening gimmemotifs' dict form."""
    f = getattr(m, "factors", None)
    if f is None:
        return []
    if isinstance(f, dict):
        out: list[str] = []
        for v in f.values():
            out.extend(v)
        return out
    return list(f)


def tiny_scan(motifs) -> None:
    """Scan a few hundred peaks end-to-end to confirm a base GRN forms with the given motifs."""
    cfg_tp = CFG["timepoints"][3]  # 15somite / TDR118reseq
    root = CFG["inputs"]["zebrahub_processed"]
    cic = CFG["inputs"]["cicero_subdir"]
    peak_csv = f"{root}/{cic}/{cfg_tp['id']}_cicero/03_{cfg_tp['id']}_processed_peak_file_danRer11.csv"
    print(f"peaks: {peak_csv}")
    peaks = pd.read_csv(peak_csv, index_col=0).head(300)
    ref_genome = CFG["scan"]["ref_genome"]
    print(f"  using {len(peaks)} peaks, ref_genome={ref_genome}")

    tfi = ma.TFinfo(peak_data_frame=peaks, ref_genome=ref_genome)
    tfi.scan(fpr=CFG["scan"]["fpr"], motifs=motifs, verbose=False, n_cpus=4)
    tfi.reset_filtering()
    tfi.filter_motifs_by_score(threshold=CFG["scan"]["motif_score_threshold"])
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=False)
    df = tfi.to_dataframe()
    print(f"  base GRN shape (peaks x cols): {df.shape}")
    tf_cols = [c for c in df.columns if c not in ("peak_id", "gene_short_name")]
    print(f"  n TF columns: {len(tf_cols)}")
    print(f"  sample TFs: {tf_cols[:15]}")
    print("  >>> tiny scan OK — end-to-end base GRN forms with the new DB")


if __name__ == "__main__":
    sys.exit(main())
