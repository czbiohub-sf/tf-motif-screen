"""Convert an AF3-derived MEME motif file into the gimmemotifs ``.pfm`` + ``.motif2factors.txt``
pair that CellOracle's motif scan consumes.

Ben's database is exported as a single MEME v4 file (``zebramotif_all_modes.meme``) where each
record looks like::

    MOTIF ZM00000000018.1 nrf1
    letter-probability matrix: alength= 4 w= 12 nsites= 10 E= 0
    0.328428  0.189443  0.205205  0.276924
    ...

The third token on the ``MOTIF`` line is the factor name. For clustered motifs it is a
*comma-joined list* of every TF that shares that motif, e.g.
``AL807792.3,ascl1a,ascl1b,atoh1a,hand2,neurog1,tal1`` — so a single motif maps to many factors.

CellOracle / gimmemotifs expect:
  * ``<name>.pfm``              — ``>MOTIF_ID`` followed by one ``A C G T`` probability row per position.
  * ``<name>.motif2factors.txt`` — TSV with columns ``Motif  Factor  Evidence  Curated``,
                                   one row per (motif, factor) pair.

This mirrors the bundled ``CisBP_ver2_Danio_rerio.{pfm,motif2factors.txt}`` exactly, so the
converted DB is a drop-in for ``tfi.scan(motifs=...)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemeMotif:
    motif_id: str
    factors: list[str]
    matrix: list[list[str]] = field(
        default_factory=list
    )  # rows of 4 prob strings, A C G T


def parse_meme(meme_path: str | Path) -> list[MemeMotif]:
    """Parse a MEME v4 file into a list of MemeMotif records.

    Robust to blank lines and the ``letter-probability matrix:`` header line. Stops a matrix
    block at the next ``MOTIF`` line, a blank line, or any non-numeric line.
    """
    meme_path = Path(meme_path)
    motifs: list[MemeMotif] = []
    current: MemeMotif | None = None
    in_matrix = False

    with meme_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("MOTIF"):
                # flush previous
                if current is not None:
                    motifs.append(current)
                parts = line.split()
                # parts = ["MOTIF", "<id>", "<names>"]  (names may be absent)
                motif_id = parts[1]
                names = parts[2] if len(parts) > 2 else ""
                factors = [f for f in names.split(",") if f]
                current = MemeMotif(motif_id=motif_id, factors=factors)
                in_matrix = False
                continue

            if current is None:
                continue

            if line.startswith("letter-probability matrix"):
                in_matrix = True
                continue

            if in_matrix:
                if not line:
                    in_matrix = False
                    continue
                tok = line.split()
                # a matrix row is 4 floats; anything else ends the block
                if len(tok) == 4 and _is_float(tok[0]):
                    current.matrix.append(tok)
                else:
                    in_matrix = False

    if current is not None:
        motifs.append(current)
    return motifs


def _is_float(s: str) -> bool:
    """Return True if the string parses as a float (used to detect matrix rows)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_pfm(motifs: list[MemeMotif], pfm_path: str | Path) -> int:
    """Write motifs to a gimmemotifs .pfm (">id" + probability rows). Returns the count written."""
    pfm_path = Path(pfm_path)
    n = 0
    with pfm_path.open("w") as fh:
        for m in motifs:
            if not m.matrix:
                continue
            fh.write(f">{m.motif_id}\n")
            for row in m.matrix:
                fh.write("\t".join(row) + "\n")
            n += 1
    return n


def write_motif2factors(
    motifs: list[MemeMotif],
    m2f_path: str | Path,
    evidence: str = "AF3",
    curated: str = "N",
) -> tuple[int, int]:
    """Write the motif2factors TSV. Returns (n_rows, n_unique_factors)."""
    m2f_path = Path(m2f_path)
    unique_factors: set[str] = set()
    n_rows = 0
    with m2f_path.open("w") as fh:
        fh.write("Motif\tFactor\tEvidence\tCurated\n")
        for m in motifs:
            if not m.matrix:
                continue
            for factor in m.factors:
                fh.write(f"{m.motif_id}\t{factor}\t{evidence}\t{curated}\n")
                unique_factors.add(factor)
                n_rows += 1
    return n_rows, len(unique_factors)


def convert(
    meme_path: str | Path, out_prefix: str | Path, evidence: str = "AF3"
) -> dict:
    """Convert a MEME file to <out_prefix>.pfm + <out_prefix>.motif2factors.txt.

    Returns a small summary dict.
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    motifs = parse_meme(meme_path)
    pfm_path = out_prefix.with_suffix(".pfm")
    m2f_path = Path(str(out_prefix) + ".motif2factors.txt")
    n_motifs = write_pfm(motifs, pfm_path)
    n_rows, n_factors = write_motif2factors(motifs, m2f_path, evidence=evidence)
    return {
        "meme": str(meme_path),
        "pfm": str(pfm_path),
        "motif2factors": str(m2f_path),
        "n_motifs_parsed": len(motifs),
        "n_motifs_written": n_motifs,
        "n_motif2factor_rows": n_rows,
        "n_unique_factors": n_factors,
    }


def main() -> None:
    """CLI entry point: convert a MEME file to the .pfm + .motif2factors.txt pair and print a summary."""
    ap = argparse.ArgumentParser(
        description="Convert AF3 MEME motifs -> gimmemotifs .pfm + .motif2factors.txt"
    )
    ap.add_argument("meme", help="input MEME file (e.g. zebramotif_all_modes.meme)")
    ap.add_argument(
        "out_prefix",
        help="output prefix; writes <prefix>.pfm and <prefix>.motif2factors.txt",
    )
    ap.add_argument(
        "--evidence",
        default="AF3",
        help="Evidence label for motif2factors (default: AF3)",
    )
    args = ap.parse_args()
    summary = convert(args.meme, args.out_prefix, evidence=args.evidence)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
