"""Build the full CisBP v3.10 danRer11 motif DB (all inferred motifs per TF) as the
gimmemotifs ``.pfm`` + ``.motif2factors.txt`` pair CellOracle's motif scan consumes.

This is the CisBP arm's motif vocabulary. It deliberately mirrors CellOracle's bundled
``CisBP_ver2_Danio_rerio.{pfm,motif2factors.txt}`` — i.e. the *all-motifs* CisBP dump where
every zebrafish TF inherits ALL of its homologs' motifs (direct + inferred). We use the v3.10
``TF_Information_all_motifs.txt`` (one row per TF x motif) rather than ``TF_Information.txt``
(one best motif per TF) so the CisBP baseline stays size-matched to the bundled v2 set instead
of shrinking ~9x — an honest, version-consistent CisBP baseline for the ZebraMotif comparison.

Not to be confused with ``data/sequences/cisbp/get_cisbp_danio.py``, which pulls the *best motif
per TF* for the AF3 benchmark. This builder is GRN-specific and keeps all motifs.

Output (paths from grn_swap.yaml ``motif_db.cisbp_prefix``):
  * ``<prefix>.pfm``               — ``>Motif_ID`` + one ``A C G T`` probability row per position.
  * ``<prefix>.motif2factors.txt`` — TSV ``Motif  Factor  Evidence  Curated`` (one row per TF-motif).

Usage:
    python -m grnswap.build_cisbp_db      # download (if absent) + build, paths from config
    python build_cisbp_db.py --force      # re-download even if the cache is present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grnswap import config as C  # noqa: E402

BASE_URL = (
    "https://cisbp.ccbr.utoronto.ca/data/{build}/DataFiles/Bulk_downloads/EntireDataset"
)
ALLMOTIFS_ZIP = "TF_Information_all_motifs.txt.zip"
PWM_ZIP = "PWMs.zip"
SPECIES = "Danio_rerio"


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Download url to dest unless it already exists (or force). Returns dest."""
    if dest.exists() and not force:
        print(f"  cached: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    print(f"  downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  done: {dest.stat().st_size/1e6:.1f} MB")
    return dest


def _read_danio_table(zip_path: Path) -> pd.DataFrame:
    """Read the all-motifs bulk table straight from its zip and filter to Danio_rerio TF-motif rows."""
    member = zip_path.name.removesuffix(".zip")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            df = pd.read_csv(fh, sep="\t", dtype=str, low_memory=False)
    dr = df[df["TF_Species"] == SPECIES].copy()
    dr = dr[dr["Motif_ID"].notna() & (dr["Motif_ID"] != ".")]
    print(
        f"  Danio rows with a motif: {len(dr)} | unique motifs: {dr['Motif_ID'].nunique()} "
        f"| unique TFs: {dr['TF_Name'].nunique()}"
    )
    return dr


def _write_pfm(dr: pd.DataFrame, pwm_zip: Path, pfm_path: Path) -> set[str]:
    """Write the gimmemotifs .pfm for every unique Danio motif that has a usable PWM.

    Reads each ``pwms/<Motif_ID>.txt`` (CisBP format: ``Pos A C G T`` header + rows) from the
    global PWM archive, drops the Pos column, and skips header-only/empty matrices. Returns the
    set of Motif_IDs actually written.
    """
    written: set[str] = set()
    with zipfile.ZipFile(pwm_zip) as zf, open(pfm_path, "w") as out:
        for motif_id in dr["Motif_ID"].drop_duplicates():
            try:
                data = zf.read(f"pwms/{motif_id}.txt").decode()
            except KeyError:
                print(f"  warning: no PWM file for {motif_id}")
                continue
            rows = []
            for ln in data.splitlines():
                parts = ln.split("\t")
                # data rows are "Pos A C G T"; skip the header and any blank/short lines
                if len(parts) == 5 and parts[0].strip().isdigit():
                    rows.append("\t".join(parts[1:]))
            if len(rows) < 1:  # header-only / empty motif — CisBP ships some of these
                print(f"  warning: empty PWM for {motif_id}, skipping")
                continue
            out.write(f">{motif_id}\n")
            out.write("\n".join(rows) + "\n")
            written.add(motif_id)
    print(f"  wrote {len(written)} motifs -> {pfm_path}")
    return written


def _write_motif2factors(dr: pd.DataFrame, kept: set[str], m2f_path: Path) -> int:
    """Write the motif2factors TSV (one row per TF-motif pair) for motifs that made it into the pfm.

    Mirrors the bundled v2 conventions: Evidence = CisBP ``MSource_Type`` (PBM/SELEX/ChIP-seq/...),
    Curated = "N". Returns the number of data rows written.
    """
    sub = dr[dr["Motif_ID"].isin(kept)]
    n = 0
    with open(m2f_path, "w") as out:
        out.write("Motif\tFactor\tEvidence\tCurated\n")
        for motif_id, factor, evidence in zip(
            sub["Motif_ID"], sub["TF_Name"], sub["MSource_Type"]
        ):
            if not isinstance(factor, str) or not factor:
                continue
            ev = evidence if isinstance(evidence, str) and evidence else "Misc"
            out.write(f"{motif_id}\t{factor}\t{ev}\tN\n")
            n += 1
    print(f"  wrote {n} motif->factor rows -> {m2f_path}")
    return n


def build_cisbp(cfg: dict, force: bool = False) -> dict:
    """Download (if needed) and build the CisBP v3.10 danRer11 gimme DB; return a summary dict."""
    mdb = cfg["motif_db"]
    build = str(mdb.get("cisbp_build", "3_10"))
    cache = Path(mdb["cisbp_cache"])
    prefix = Path(mdb["cisbp_prefix"])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base = BASE_URL.format(build=build)

    print(f"=== Build CisBP v{build} {SPECIES} all-motifs DB ===")
    tbl_zip = _download(f"{base}/{ALLMOTIFS_ZIP}", cache / ALLMOTIFS_ZIP, force)
    pwm_zip = _download(f"{base}/{PWM_ZIP}", cache / PWM_ZIP, force)

    dr = _read_danio_table(tbl_zip)
    # keep the raw danio subset for provenance
    dr.to_csv(cache / f"TF_Information_all_motifs_{SPECIES}.txt", sep="\t", index=False)

    pfm_path = Path(str(prefix) + ".pfm")
    m2f_path = Path(str(prefix) + ".motif2factors.txt")
    kept = _write_pfm(dr, pwm_zip, pfm_path)
    n_rows = _write_motif2factors(dr, kept, m2f_path)

    factors = set(dr[dr["Motif_ID"].isin(kept)]["TF_Name"].dropna())
    return {
        "build": build,
        "pfm": str(pfm_path),
        "motif2factors": str(m2f_path),
        "n_motifs": len(kept),
        "n_factors": len(factors),
        "n_m2f_rows": n_rows,
    }


def main() -> None:
    """Build the CisBP DB from config and print the summary."""
    ap = argparse.ArgumentParser(
        description="Build the CisBP v3.10 danRer11 gimme motif DB"
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    s = build_cisbp(cfg, force=args.force)
    for k, v in s.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
