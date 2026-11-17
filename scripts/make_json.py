"""
Creates JSON files for AlphaFold3 jobs.

Ben Iovino  12/13/24    CZ-Biohub
"""

import argparse
import glob
import multiprocessing as mp
import os
import sqlite3
import types
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from af3tfm.dataset import DS
from af3tfm.jobdb import Database
from af3tfm.motif import Motif
from af3tfm.tfactor import TFactor, load_tfs


def read_top_motifs(path: str) -> dict:
    """Reads a file with the top motifs for each TF.

    Args:
        path (str): Path to the file.

    Returns:
        dict: Dictionary with TF DBID as key and list of top motifs as value.
    """

    top_motifs = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip().split(",")
            top_motifs[line[0]] = line[1:]

    return top_motifs


def read_fasta(path: str) -> tuple[list[str], list[str]]:
    """Reads a fasta file.

    Args:
        path (str): Path to the fasta file.

    Returns:
        tuple[list[str], list[str]]: (headers, sequences) in file order.
    """

    headers, seqs = [], []
    with open(path, "r") as f:
        header, chunks = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    headers.append(header)
                    seqs.append("".join(chunks))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
        if header is not None:
            headers.append(header)
            seqs.append("".join(chunks))

    return headers, seqs


def build_negative_jobs(neg_dir: str, dataset, data_path: str, job_dir: str):
    """Create negative GPU jobs from a directory of <TF>_negatives.fa fastas.

    Args:
        neg_dir (str): Directory of <TF>_negatives.fa files.
        dataset: Dataset object providing the TF sequence CSV.
        data_path (str): Path to the reused protein MSA/template data.
        job_dir (str): Directory to write job JSONs.
    """

    # Key by DBID (unambiguous); fall back to TF_Name for backward compatibility
    tf_df = pd.read_csv(dataset.seq_csv)
    tfs = load_tfs(tf_df, "DBID", data_path, database=None)
    lookup = {dbid.lower(): tf for dbid, tf in tfs.items()}
    for tf in tfs.values():
        lookup.setdefault(tf.name.lower(), tf)

    neg_files = sorted(glob.glob(os.path.join(neg_dir, "*_negatives.fa")))
    print(f"Found {len(neg_files)} TFs with negatives")

    for neg_file in tqdm(neg_files, desc="Processing negatives"):
        key = os.path.basename(neg_file)[: -len("_negatives.fa")]

        if key.lower() not in lookup:
            print(f"  WARNING: {key} not in {dataset.name}, skipping")
            continue
        tf = lookup[key.lower()]

        headers, seqs = read_fasta(neg_file)
        tf.motif = Motif(name=None, motif_seqs=seqs, seq_names=headers)
        tf.create_af_job("gpu", job_dir)


def _process_motif_worker(args: tuple):
    """
    Worker: generate sequences for one motif and write all job JSONs.

    Each worker opens its own read-only SQLite connection so concurrent
    processes do not share a connection across fork boundaries.
    """

    pwm_file_str, tf_records, top_motifs, motif_num, job_dir, db_path = args

    pwm_file_path = Path(pwm_file_str)
    motif = Motif(name=pwm_file_path.stem, pwm_path=pwm_file_str)
    motif.generate_sequences(motif_num)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) if db_path else None
    cursor = conn.cursor() if conn else None
    db_obj = types.SimpleNamespace(cursor=cursor) if cursor else None

    try:
        for tf_name, ensembl_id, sequence, data_path in tf_records:
            if top_motifs:
                key = ensembl_id.lower()
                if key not in top_motifs or motif.name not in top_motifs[key]:
                    continue

            tf = TFactor(
                name=tf_name,
                ensembl_ID=ensembl_id,
                sequence=sequence,
                data_path=data_path,
                motif=motif,
                db=db_obj,
            )
            tf.create_af_job("gpu", job_dir)
    finally:
        if conn:
            conn.close()


def main():
    """ """

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, help="Type of job (cpu or gpu).")
    parser.add_argument("--job_dir", type=str, help="Path to the directory of jobs.")
    parser.add_argument("--dataset", type=str, help="Dataset to use.")
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data directory (if gpu job)",
        default=None,
    )
    parser.add_argument("--motif_num", type=int, help="Number of motifs to generate.")
    parser.add_argument(
        "--top_motifs",
        type=str,
        help="Path to the top motifs file (if gpu job)",
        default=None,
    )
    parser.add_argument(
        "--pwm_path",
        type=str,
        help="Path to the position weight matrices (if gpu job)",
        default=None,
    )
    parser.add_argument(
        "--database_path",
        type=str,
        help="Path to the job database",
        default=None,
    )
    parser.add_argument(
        "--neg_dir",
        type=str,
        help="Directory of <TF>_negatives.fa fastas of non-motif sequences "
        "(gpu job). If set, one negative job is created per fasta sequence, "
        "in addition to the positive jobs sampled from the PWMs.",
        default=None,
    )
    args = parser.parse_args()

    dataset = DS[args.dataset]
    print(f"Dataset: {dataset.name}")

    db = Database(
        job_path=Path(args.job_dir).parent,
        dataset=dataset.name,
    )

    tf_df = pd.read_csv(dataset.seq_csv)
    tfs = load_tfs(tf_df, "DBID", args.data_path, db)

    # For CPU, we create jobs with only proteins
    if args.job_type == "cpu":
        for tf in tfs:
            tfs[tf].create_af_job(args.job_type, args.job_dir)

    # For GPU, we create jobs with proteins and DNA sequences
    elif args.job_type == "gpu":
        if args.top_motifs:
            top_motifs = read_top_motifs(args.top_motifs)
        else:
            top_motifs = None
        pwm_path = Path(args.pwm_path)
        motif_files = [p for p in pwm_path.iterdir() if p.suffix in (".motif", ".txt")]

        # Serialize TF data to worker processes
        tf_records = [
            (tfs[tf].name, tfs[tf].ensembl_ID, tfs[tf].sequence, tfs[tf].data_path)
            for tf in tfs
        ]
        db_path = str(db.db_path)
        db.conn.close()

        worker_args = [
            (
                str(f),
                tf_records,
                top_motifs,
                args.motif_num,
                args.job_dir,
                db_path,
            )
            for f in motif_files
        ]

        with mp.Pool(processes=os.cpu_count()) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_motif_worker, worker_args, chunksize=1),
                total=len(motif_files),
                desc="Processing motifs",
            ):
                pass

        # Negative jobs: genomic non-motif sequences read from fastas
        if args.neg_dir:
            build_negative_jobs(args.neg_dir, dataset, args.data_path, args.job_dir)


if __name__ == "__main__":
    main()
