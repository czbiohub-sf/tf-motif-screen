"""
Build the curated CisBP zebrafish TF table (cisbp_tfs.csv) by slicing the
hand-picked TFs out of the full Danio_rerio TF_Information.txt retrieved by
get_cisbp_danio.py, fetching each canonical protein sequence, and annotating the
AnimalTFDB4 family. Each TF's Motif_ID (and thus its pwms/<Motif_ID>.txt PWM)
comes straight from that table.

Ben Iovino  2/03/25    CZ-Biohub
"""

import sys
from pathlib import Path

import pandas as pd

from af3tfm.dataset import DATA_DIR

sys.path.append(str(Path(__file__).resolve().parents[2]))
from get_seqs import fetch_canonical_protein  # noqa: E402

# Hand-picked zebrafish TFs. The first group are direct-evidence CisBP motifs;
# the second are added for DNA-binding-domain diversity (high homology to human
# proteins with JASPAR motifs) and now also carry inferred CisBP motifs.
CURATED_TFS = [
    "arid6",
    "phf21ab",
    "kdm2aa",
    "kdm2ba",
    "dharma",
    "hoxd13a",
    "si:dkey-43p13.5",
    "vsx2",
    "pou3f3b",
    "hsf2",
    "myrf",
    "nr2f2",
    "pax2b",
    "lin54",
    "prkrira",
    "prkrirb",
    "gata3",
    "sox2",
    "pax2a",
    "jund",
    "myod1",
    "klf3",
    "foxo1a",
    "tbx1",
    "tbx2a",
    "stat3",
    "mef2aa",
    "sp2",
    "erg",
]


def add_family(tf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the AnimalTFDB4 family annotation for each TF, matched on gene ID.

    Args:
        tf_df (pd.DataFrame):
            Cleaned TF dataframe.

    Returns:
        pd.DataFrame:
            TF dataframe with a Family column (empty for TFs absent from AnimalTFDB4).
    """

    tfdb_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
    families = dict(zip(tfdb_df["DBID"], tfdb_df["Family"]))
    tf_df.insert(
        tf_df.columns.get_loc("Sequence"), "Family", tf_df["DBID"].map(families)
    )

    return tf_df


def main():
    """ """

    info = pd.read_csv(
        f"{DATA_DIR}/sequences/cisbp/TF_Information.txt", sep="\t", dtype=str
    )
    tf_df = (
        info[info["TF_Name"].isin(CURATED_TFS)]
        .drop_duplicates("TF_Name")[["TF_Name", "DBID", "Motif_ID"]]
        .copy()
    )
    tf_df["Sequence"] = tf_df["DBID"].apply(fetch_canonical_protein)
    tf_df = tf_df[tf_df["Sequence"].notna()]
    tf_df = add_family(tf_df)
    tf_df.to_csv(f"{DATA_DIR}/sequences/cisbp/cisbp_tfs.csv", index=False)


if __name__ == "__main__":
    main()
