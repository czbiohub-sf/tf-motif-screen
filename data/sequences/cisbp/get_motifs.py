"""Assemble the per-TF motif PWMs for the motif/non-motif experiment into a
single `motifs/` directory, kept separate from the raw CisBP `pwms/` dump.

Outputs:
  motifs/<motif_id>.txt   one PWM per TF (CisBP for direct, JASPAR for indirect)
  motifs.txt              "dbid,tf_name,motif_id" per TF (make_json --top_motifs)

Ben Iovino  9/4/2026    CZ-Biohub
"""

import shutil

import pandas as pd

from af3tfm.dataset import DATA_DIR

HERE = DATA_DIR / "sequences/cisbp"
CISBP_TFS = HERE / "cisbp_tfs.csv"
TF_INFO = HERE / "TF_Information.txt"
PWM_DIR = HERE / "pwms"
REP_TFS = DATA_DIR / "sequences/animaltfdb/tfdb4_rep_tfs.csv"
JASPAR_MEME = DATA_DIR / "tools/homer/jaspar.meme.motif"

MOTIF_DIR = HERE / "motifs"
MANIFEST = HERE / "motifs.txt"

# JASPAR IDs for inferred TFs that are absent from the AnimalTFDB4 rep table
# (e.g. clustered under a different representative), keyed by gene symbol
JASPAR_OVERRIDES = {
    "klf3": "MA1516.1",  # not a rep in tfdb4_rep_tfs.csv
}


def read_meme(path) -> dict[str, list[list[float]]]:
    """Parse a MEME-format motif library into {motif_id: [[A, C, G, T], ...]}.

    Each `MOTIF <id> <name>` block is followed by a letter-probability matrix of
    one row per position (A C G T order). Only 4-float matrix rows are kept, so
    the header, `letter-probability`, `URL`, and blank lines are ignored.

    Args:
        path: Path to the MEME motif file.

    Returns:
        dict: Motif ID -> list of per-position [A, C, G, T] probabilities.
    """

    motifs: dict[str, list[list[float]]] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF "):
                current = line.split()[1]
                motifs[current] = []
                continue
            if current is None:
                continue
            parts = line.split()
            if len(parts) == 4:
                try:
                    motifs[current].append([float(x) for x in parts])
                except ValueError:
                    pass

    return motifs


def write_pwm(rows: list[list[float]], dest) -> None:
    """Write a probability matrix in the CisBP tab format (Pos A C G T)."""

    with open(dest, "w") as fh:
        fh.write("Pos\tA\tC\tG\tT\n")
        for i, (a, c, g, t) in enumerate(rows, start=1):
            fh.write(f"{i}\t{a}\t{c}\t{g}\t{t}\n")


def main() -> None:
    cis = pd.read_csv(CISBP_TFS, dtype=str)
    info = pd.read_csv(TF_INFO, sep="\t", dtype=str)
    rep = pd.read_csv(REP_TFS, dtype=str)

    # CisBP evidence level per TF, matched on the exact (gene, motif) pair
    status = info.drop_duplicates(["DBID", "Motif_ID"]).set_index(["DBID", "Motif_ID"])[
        "TF_Status"
    ]
    # Homologous human JASPAR matrix per zebrafish gene (AnimalTFDB4 reps),
    # keyed on gene symbol -- Ensembl gene IDs drift between CisBP and
    # AnimalTFDB4 (e.g. tbx2a), so a DBID join silently misses those TFs
    rep_uniq = rep.dropna(subset=["Hom_JASPAR"]).drop_duplicates("TF_Name")
    jaspar_by_name = dict(zip(rep_uniq["TF_Name"].str.lower(), rep_uniq["Hom_JASPAR"]))
    meme = read_meme(JASPAR_MEME)

    MOTIF_DIR.mkdir(exist_ok=True)
    manifest: list[tuple[str, str, str]] = []
    for row in cis.itertuples():
        tf_status = status.get((row.DBID, row.Motif_ID))

        if tf_status == "D":  # direct evidence: use the TF's own CisBP PWM
            src = PWM_DIR / f"{row.Motif_ID}.txt"
            if not src.exists():
                print(f"  warning: no CisBP PWM for {row.TF_Name} ({row.Motif_ID})")
                continue
            motif_id = row.Motif_ID
            shutil.copyfile(src, MOTIF_DIR / f"{motif_id}.txt")

        else:  # inferred evidence: use the homolog's JASPAR PWM
            name = row.TF_Name.lower()
            jaspar_id = JASPAR_OVERRIDES.get(name) or jaspar_by_name.get(name)
            if not jaspar_id:
                print(
                    f"  warning: {row.TF_Name} ({row.DBID}) inferred but has no "
                    "Hom_JASPAR homolog, skipping"
                )
                continue
            if jaspar_id not in meme:
                print(
                    f"  warning: JASPAR {jaspar_id} for {row.TF_Name} not in "
                    f"{JASPAR_MEME.name}, skipping"
                )
                continue
            motif_id = jaspar_id
            write_pwm(meme[jaspar_id], MOTIF_DIR / f"{motif_id}.txt")

        manifest.append((row.DBID.lower(), row.TF_Name, motif_id))

    with open(MANIFEST, "w") as fh:
        for dbid, tf_name, motif_id in manifest:
            fh.write(f"{dbid},{tf_name},{motif_id}\n")


if __name__ == "__main__":
    main()
