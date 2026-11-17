"""Stage 02: compute the base GRN (peaks x TFs) from a timepoint's stage-03 peaks, scanning
with the chosen motif DB (CisBP v3.10, ZebraMotif, or hybrid).

Vendored from zebrahub-multiome-analysis/scripts/preprocessing/run_04_celloracle_compute_baseGRN.py.
The ONLY substantive change is the motif source: `tfi.scan(motifs=<chosen DB>)` instead of
`motifs=None`. All scan params (fpr, motif_score_threshold, ref_genome) come from grn_swap.yaml
and are identical across arms, so the base GRN differs only by motif vocabulary.

Usage:
    python 02_compute_base_grn.py --index 3 --db cisbp       # one timepoint by config index
    python 02_compute_base_grn.py --index $SLURM_ARRAY_TASK_ID --db zebramotif
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from celloracle import motif_analysis as ma
from genomepy import Genome
from gimmemotifs.motif import read_motifs as _gimme_read_motifs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grnswap import config as C  # noqa: E402


def load_motifs(pfm_path: str):
    """Load motifs with factor info, preferring CellOracle's reader over gimmemotifs.

    Both readers auto-read the sibling .motif2factors.txt.
    """
    if hasattr(ma, "read_motifs"):
        return ma.read_motifs(pfm_path)
    return _gimme_read_motifs(pfm_path)


def decompose_chrstr(peak_str: str):
    """Split a "chr_start_end" peak id into (chr, start, end), tolerating "_" inside the chr name."""
    *chr_, start, end = peak_str.split("_")
    return "_".join(chr_), start, end


def check_peak_format(peaks_df: pd.DataFrame, ref_genome: str) -> pd.DataFrame:
    """Drop peaks with an unknown chromosome or a length under 5 bp; report how many were removed."""
    df = peaks_df.copy()
    n_before = df.shape[0]
    decomposed = [decompose_chrstr(p) for p in df["peak_id"]]
    d = pd.DataFrame(
        np.array(decomposed), index=peaks_df.index, columns=["chr", "start", "end"]
    )
    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)
    all_chr = list(Genome(ref_genome).keys())
    lengths = np.abs(d["end"] - d["start"])
    df = df[(lengths >= 5) & d.chr.isin(all_chr)]
    print(
        f"Peaks before: {n_before} | invalid chr: {n_before - d.chr.isin(all_chr).sum()} | "
        f"invalid length: {int((lengths < 5).sum())} | after: {df.shape[0]}"
    )
    return df


def main() -> None:
    """Scan one timepoint's peaks with the chosen motif DB and write the base GRN parquet."""
    ap = argparse.ArgumentParser(
        description="Compute base GRN with the chosen motif DB (cisbp | zebramotif | hybrid)"
    )
    ap.add_argument("--config", default=None)
    C.add_annotation_argument(ap)
    ap.add_argument(
        "--index", type=int, required=True, help="timepoint index in grn_swap.yaml"
    )
    ap.add_argument(
        "--db",
        choices=["cisbp", "zebramotif", "hybrid"],
        default="zebramotif",
        help="which motif DB to scan with",
    )
    ap.add_argument(
        "--motifs",
        default=None,
        help="override motif .pfm path (default: from config by --db)",
    )
    args = ap.parse_args()

    cfg = C.load_config(args.config, annotation=args.annotation)
    tp = C.timepoint(cfg, args.index)
    ref_genome = cfg["scan"]["ref_genome"]
    pfm_path = args.motifs or C.motif_pfm_for_db(cfg, args.db)

    print(
        f"=== Stage 02 ({args.db.upper()}) | {tp['stage']} / {tp['id']} | motifs={pfm_path} ==="
    )
    print(
        f"ref_genome={ref_genome} (installed={ma.is_genome_installed(ref_genome=ref_genome)})"
    )

    peaks = pd.read_csv(C.peak_csv(cfg, tp), index_col=0)
    motifs = load_motifs(pfm_path)

    tfi = ma.TFinfo(peak_data_frame=peaks, ref_genome=ref_genome)
    peaks = check_peak_format(
        peaks, ref_genome
    )  # (matches original ordering/behaviour)

    t0 = time.time()
    tfi.scan(
        fpr=cfg["scan"]["fpr"],
        motifs=motifs,
        verbose=True,
        n_cpus=cfg["scan"]["n_cpus"],
    )
    print(f"motif scan: {(time.time()-t0)/60:.1f} min")

    tfi.to_hdf5(file_path=C.tfinfo_out(cfg, tp, args.db))
    tfi.reset_filtering()
    tfi.filter_motifs_by_score(threshold=cfg["scan"]["motif_score_threshold"])
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)

    df = tfi.to_dataframe()
    out = C.base_grn_db(cfg, tp, args.db)
    df.to_parquet(out)
    tf_cols = [c for c in df.columns if c not in ("peak_id", "gene_short_name")]
    print(f"base GRN: {df.shape} | n TFs: {len(tf_cols)} | saved -> {out}")


if __name__ == "__main__":
    main()
