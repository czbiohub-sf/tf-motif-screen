"""Build a union-hybrid motif DB: concatenate the ZebraMotif and CisBP gimmemotifs .pfm + .motif2factors.

Union semantics: every TF keeps BOTH its ZebraMotif and CisBP motif(s). CellOracle aggregates
multiple motifs per factor with OR, so the hybrid candidate-edge set is a strict superset of the
cisbp ∪ zebramotif arms — CisBP retains all canonical edges and ZebraMotif adds the rescued TFs.
Motif IDs are disjoint (M* vs ZM*), so this is a literal concatenation of both DBs.

Usage:
    python -m grnswap.build_hybrid_db        # paths from grn_swap.yaml
    python build_hybrid_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grnswap import config as C  # noqa: E402


def _read_pfm_ids(pfm: str) -> list[str]:
    """Return the motif ids (">" header lines) from a gimmemotifs .pfm file."""
    return [
        ln[1:].strip()
        for ln in Path(pfm).read_text().splitlines()
        if ln.startswith(">")
    ]


def _concat_text(paths: list[str], out_path: str) -> None:
    """Concatenate text files verbatim into out_path, ensuring each ends with a newline."""
    with open(out_path, "w") as out:
        for p in paths:
            txt = Path(p).read_text()
            if not txt.endswith("\n"):
                txt += "\n"
            out.write(txt)


def _concat_motif2factors(paths: list[str], out_path: str) -> int:
    """Concatenate motif2factors files keeping a single header. Returns total data rows."""
    n = 0
    with open(out_path, "w") as out:
        out.write("Motif\tFactor\tEvidence\tCurated\n")
        for p in paths:
            for i, line in enumerate(Path(p).read_text().splitlines()):
                if i == 0 and line.lower().startswith("motif\t"):
                    continue  # skip per-file header
                if line.strip():
                    out.write(line + "\n")
                    n += 1
    return n


def build_hybrid(cfg: dict) -> dict:
    """Concatenate the CisBP and ZebraMotif DBs into the hybrid .pfm + .motif2factors and return a summary."""
    zebramotif_pfm = cfg["motif_db"]["zebramotif_prefix"] + ".pfm"
    zebramotif_m2f = cfg["motif_db"]["zebramotif_prefix"] + ".motif2factors.txt"
    cisbp_pfm = cfg["motif_db"]["cisbp_prefix"] + ".pfm"
    cisbp_m2f = cfg["motif_db"]["cisbp_prefix"] + ".motif2factors.txt"
    hyb = cfg["motif_db"]["hybrid_prefix"]
    hyb_pfm, hyb_m2f = hyb + ".pfm", hyb + ".motif2factors.txt"
    Path(hyb).parent.mkdir(parents=True, exist_ok=True)

    # order: CisBP first then ZebraMotif (arbitrary; IDs disjoint)
    _concat_text([cisbp_pfm, zebramotif_pfm], hyb_pfm)
    n_rows = _concat_motif2factors([cisbp_m2f, zebramotif_m2f], hyb_m2f)

    cisbp_ids, zebramotif_ids = (
        set(_read_pfm_ids(cisbp_pfm)),
        set(_read_pfm_ids(zebramotif_pfm)),
    )
    hyb_ids = _read_pfm_ids(hyb_pfm)

    def factors(m2f):
        """Collect the unique factor names (column 2) from a motif2factors file."""
        s = set()
        for i, line in enumerate(Path(m2f).read_text().splitlines()):
            if i == 0:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1]:
                s.add(parts[1])
        return s

    f_cisbp, f_zebramotif = factors(cisbp_m2f), factors(zebramotif_m2f)
    return {
        "hybrid_pfm": hyb_pfm,
        "hybrid_m2f": hyb_m2f,
        "n_motifs_cisbp": len(cisbp_ids),
        "n_motifs_zebramotif": len(zebramotif_ids),
        "n_motifs_hybrid": len(hyb_ids),
        "id_collisions": len(cisbp_ids & zebramotif_ids),
        "n_m2f_rows": n_rows,
        "n_factors_cisbp": len(f_cisbp),
        "n_factors_zebramotif": len(f_zebramotif),
        "n_factors_union": len(f_cisbp | f_zebramotif),
        "n_factors_shared": len(f_cisbp & f_zebramotif),
    }


def main() -> None:
    """Build the hybrid DB from config, print the summary, and assert IDs/counts are consistent."""
    cfg = C.load_config()
    s = build_hybrid(cfg)
    for k, v in s.items():
        print(f"{k}: {v}")
    assert s["id_collisions"] == 0, "motif-id collision between CisBP and ZebraMotif!"
    assert (
        s["n_motifs_hybrid"] == s["n_motifs_cisbp"] + s["n_motifs_zebramotif"]
    ), "motif count mismatch"


if __name__ == "__main__":
    main()
