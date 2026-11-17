"""
Reads a downloaded or directly accessible AnimalTFDB4 transcription factor
list and fetches each listed protein sequence from Ensembl.

Ben Iovino  12/8/25    CZ-Biohub
"""

import io
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

from af3tfm.dataset import DATA_DIR

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)  # data/get_seqs.py, not this file
from get_seqs import fetch_proteins  # noqa: E402


def download_fasta(url: str) -> str:
    """Fetch a directly accessible FASTA once, without access-control workarounds.

    If AnimalTFDB requires browser interaction, download the file manually and
    pass its local path with ``--input``.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_fasta(fasta: str) -> pd.DataFrame:
    """Parses the AnimalTFDB4 fasta headers to a dataframe of TFs.

    Only the headers are used; sequences are fetched from Ensembl.

    Args:
        fasta (str): The AnimalTFDB4 fasta file contents.

    Returns:
        pd.DataFrame: Protein ID and version, gene ID, name, and family per TF.
    """

    if not fasta.lstrip().startswith(">"):
        raise ValueError(
            "Expected an AnimalTFDB protein FASTA file. Download it in your "
            "browser and pass the saved file with --input."
        )

    pids, versions, dbids, names, fams = [], [], [], [], []
    for seq in SeqIO.parse(io.StringIO(fasta), "fasta"):
        desc = seq.description.split()
        try:
            pid, dbid, name, fam = desc[0], desc[1], desc[2], desc[3]
        except IndexError:  # no gene name
            continue
        pid, _, version = pid.partition(".")
        pids.append(pid)
        versions.append(version)
        dbids.append(dbid)
        names.append(name)
        fams.append(fam)

    if not pids:
        raise ValueError(
            "No AnimalTFDB headers contain protein ID, gene ID, gene name, "
            "and family. Check the downloaded protein FASTA file."
        )

    tf_df = pd.DataFrame(
        {
            "ProteinID": pids,
            "Version": versions,
            "DBID": dbids,
            "TF_Name": names,
            "Family": fams,
        }
    )

    return tf_df


def add_sequences(tf_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the Ensembl protein sequence of each TF.

    TFs whose protein Ensembl no longer returns are dropped, and a version that
    differs from the one AnimalTFDB4 lists is reported.

    Args:
        tf_df (pd.DataFrame): TF dataframe from parse_fasta.

    Returns:
        pd.DataFrame: TF dataframe with a Sequence column.
    """

    proteins = fetch_proteins(tf_df["ProteinID"].tolist())
    for row in tf_df.itertuples():
        if row.ProteinID not in proteins:
            print(f"Warning: {row.ProteinID} ({row.TF_Name}) not found in Ensembl.")
        elif (
            proteins[row.ProteinID][0] is not None
            and str(proteins[row.ProteinID][0]) != row.Version
        ):
            print(
                f"Warning: {row.ProteinID} is version {proteins[row.ProteinID][0]} "
                f"in Ensembl, AnimalTFDB4 lists version {row.Version}."
            )

    tf_df["Sequence"] = tf_df["ProteinID"].map(
        lambda pid: proteins[pid][1] if pid in proteins else None
    )
    tf_df["Version"] = tf_df["ProteinID"].map(
        lambda pid: str(proteins[pid][0])
        if pid in proteins and proteins[pid][0] is not None
        else ""
    )
    tf_df = tf_df[tf_df["Sequence"].notna()]

    return tf_df


def write_fasta(tf_df: pd.DataFrame, filename: str):
    """Writes TF sequences to a fasta file with AnimalTFDB4-style headers.

    Args:
        tf_df (pd.DataFrame): TF dataframe with sequences.
        filename (str): The name of the fasta file to write.
    """

    with open(filename, "w") as f:
        for row in tf_df.itertuples():
            version = f".{row.Version}" if row.Version else ""
            f.write(
                f">{row.ProteinID}{version}\t{row.DBID}\t{row.TF_Name}\t{row.Family}\n"
            )
            for i in range(0, len(row.Sequence), 60):
                f.write(f"{row.Sequence[i : i + 60]}\n")


def main():
    """Build the TF catalog from AnimalTFDB headers and Ensembl sequences."""

    parser = ArgumentParser(
        description="Read an AnimalTFDB4 TF list and fetch sequences from Ensembl."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        help="AnimalTFDB protein FASTA downloaded manually in your browser.",
    )
    source.add_argument(
        "--url",
        type=str,
        default="https://guolab.wchscu.cn/AnimalTFDB4_static/download/TF_list_protein_list_final/Danio_rerio_TF_protein.fasta",
        help="Direct FASTA URL; use --input if the site requires browser interaction.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.fasta",
    )
    args = parser.parse_args()

    try:
        fasta = args.input.read_text() if args.input else download_fasta(args.url)
        tf_df = parse_fasta(fasta)
    except (OSError, ValueError, requests.RequestException) as error:
        parser.error(
            f"{error} Use --input with a manually downloaded AnimalTFDB FASTA."
        )
    tf_df = add_sequences(tf_df)
    write_fasta(tf_df, args.filename)
    tf_df_filename = args.filename.replace(".fasta", ".csv")
    tf_df[["DBID", "TF_Name", "Family", "Sequence"]].to_csv(tf_df_filename, index=False)


if __name__ == "__main__":
    main()
