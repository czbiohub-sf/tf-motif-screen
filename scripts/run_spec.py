"""
Generates and submits a SLURM array job running NA-MPNN specificity prediction
for each TF present in both the dataset CSV and the job database.

Ben Iovino  4/21/26  Biohub
"""

import argparse
import sqlite3
import subprocess as sp
import sys
from pathlib import Path

import pandas as pd
from af3tfm.config import CONFIG
from af3tfm.dataset import DS

_PYTHON = sys.executable


def get_proteins(
    dataset_name: str, jobdb_path: Path, proteins_file: Path | None = None
) -> list[str]:
    """
    Return protein IDs present in both the dataset CSV and the job database.

    Args:
        dataset_name (str): Key into the DS dict.
        jobdb_path (Path): Path to jobdb.sqlite.
        proteins_file (Path | None): If given, a newline-delimited file of DBIDs
            (case-insensitive) to further restrict the result to.

    Returns:
        proteins (list[str]): Sorted list of matching protein IDs.
    """
    dataset = DS[dataset_name]
    tf_df = pd.read_csv(dataset.seq_csv)
    csv_proteins = set(tf_df["DBID"].str.lower())

    conn = sqlite3.connect(str(jobdb_path))
    db_proteins = {
        row[0] for row in conn.execute("SELECT DISTINCT protein FROM runs").fetchall()
    }
    conn.close()

    proteins = csv_proteins & db_proteins
    if proteins_file is not None:
        restrict = {
            line.strip().lower()
            for line in Path(proteins_file).read_text().splitlines()
            if line.strip()
        }
        proteins &= restrict

    return sorted(proteins)


def get_all_structs(
    proteins: list[str],
    jobdb_path: Path,
    n_structs: int = 50,
    fold_dirs: list[str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """
    Return (protein, motif, dna) tuples for the top n_structs per protein.

    Uses the same combined percentile rank as Specificity._get_top_structs.

    Args:
        proteins (list[str]): Protein IDs to query.
        jobdb_path (Path): Path to jobdb.sqlite.
        n_structs (int): Number of top structures per protein.
        fold_dirs (list[str] | None): If given, only consider structures whose
            path lies under one of these AF3 output directory names (e.g.
            "output_fold_all"). The percentile ranking is then computed over
            the filtered set only.

    Returns:
        structs (list[tuple[str, str, str, str]]): (protein, motif, dna, seed) tuples.
    """
    placeholders = ",".join("?" * len(proteins))
    where = f"protein IN ({placeholders})"
    params = list(proteins)
    if fold_dirs:
        # Trailing slash keeps "output_fold_all" from also matching "output_fold_all2".
        like_clauses = " OR ".join("structure_path LIKE ?" for _ in fold_dirs)
        where += f" AND ({like_clauses})"
        params += [f"%/{d}/%" for d in fold_dirs]

    conn = sqlite3.connect(str(jobdb_path))
    df = pd.read_sql(
        f"SELECT protein, dna, ipTM, dom_pLDDT, motif, structure_path FROM runs WHERE {where}",
        conn,
        params=params,
    )
    conn.close()

    df["combined"] = df.groupby("protein")["ipTM"].rank(pct=True) * df.groupby(
        "protein"
    )["dom_pLDDT"].rank(pct=True)
    top = df.sort_values("combined", ascending=False).groupby("protein").head(n_structs)
    return [
        (
            row.protein,
            row.motif,
            row.dna,
            Path(row.structure_path).parent.parent.name.rsplit("_", 1)[-1],
        )
        for row in top.itertuples()
    ]


def write_job_list(
    structs: list[tuple[str, str, str, str]], out_dir: Path, tag: str = ""
) -> Path:
    """
    Write one tab-separated protein/motif/dna/seed entry per line for the SLURM array.

    Args:
        structs (list[tuple[str, str, str, str]]): (protein, motif, dna, seed) tuples.
        out_dir (Path): Directory to write the file into.
        tag (str): Optional suffix so concurrent submissions (e.g. a parameter
            sweep) don't clobber each other's job list. The SLURM array reads
            this file lazily at runtime, so it must stay stable per submission.

    Returns:
        job_list (Path): Path to the written file.
    """
    job_list = out_dir / (f"job_names_{tag}.txt" if tag else "job_names.txt")
    with open(job_list, "w") as f:
        for protein, motif, dna, seed in structs:
            f.write(f"{protein}\t{motif}\t{dna}\t{seed}\n")

    return job_list


def generate_slurm_script(
    job_list: Path,
    n_jobs: int,
    jobdb_dir: Path,
    dataset_name: str,
    partition: str,
    spec_dir: str,
    n_batches: int = 30,
    softmax_temp: float = 0.1,
    gpus: int = -1,
    design_na_only: bool = False,
) -> str:
    """
    Generate a SLURM array script that runs the worker for each protein.

    Args:
        job_list (Path): Path to the file containing protein IDs.
        n_jobs (int): Number of array tasks.
        jobdb_dir (Path): Parent directory of jobdb.sqlite.
        dataset_name (str): Dataset key (forwarded to the worker).
        partition (str): SLURM partition.
        spec_dir (str): Subdirectory of jobdb_dir for specificity outputs.
        n_batches (int): NA-MPNN forward passes per structure (--number_of_batches).
        softmax_temp (float): NA-MPNN sampling temperature (--temperature).
        gpus (int): GPUs to request. -1 (auto) requests 1 on gpu/preempted, else 0.
            NA-MPNN specificity runs on CPU, so pass 0 to use a partition's
            CPU-only nodes (e.g. --partition preempted --gpus 0).
        design_na_only (bool): Fix protein chains so NA-MPNN sees amino acid identities.

    Returns:
        script (str): The SLURM script as a string.
    """
    logs_dir = jobdb_dir / "slurm_logs"

    script = f"""#!/bin/bash
#SBATCH --job-name=na-mpnn
#SBATCH --time=0:20:00
#SBATCH --mem=8G
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL
#SBATCH --output={logs_dir}/%j.log
#SBATCH --error={logs_dir}/%j.err
#SBATCH --array=1-{n_jobs}
"""
    if CONFIG["mail_user"]:
        script += f"#SBATCH --mail-user={CONFIG['mail_user']}\n"
    n_gpus = 1 if (gpus == -1 and partition in ("gpu", "preempted")) else max(gpus, 0)
    if n_gpus > 0:
        script += f"#SBATCH --gpus={n_gpus}\n"

    script += f"""

export NA_MPNN_N_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
JOB_LIST="{job_list}"
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$JOB_LIST")
PROTEIN=$(echo "$LINE" | cut -f1)
MOTIF=$(echo "$LINE" | cut -f2)
DNA=$(echo "$LINE" | cut -f3)
SEED=$(echo "$LINE" | cut -f4)

{_PYTHON} - <<EOF
from af3tfm.jobdb import Database
from af3tfm.specificity import Specificity
db = Database(job_path="{jobdb_dir}", dataset="{dataset_name}")
Specificity(db=db, protein="$PROTEIN", out_subdir="{spec_dir}", n_batches={n_batches}, softmax_temp={softmax_temp}, design_na_only={design_na_only}).predict(motif="$MOTIF", dna="$DNA", seed="$SEED")
db.conn.close()
EOF
"""
    return script


def main():
    """ """

    parser = argparse.ArgumentParser(
        description="Run NA-MPNN specificity prediction for each TF in a job database."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument(
        "--jobdb",
        type=str,
        required=True,
        help="Path to jobdb.sqlite.",
    )
    parser.add_argument("--partition", type=str, default="cpu", help="SLURM partition.")
    parser.add_argument(
        "--n-structs", type=int, default=50, help="Top structures per TF."
    )
    parser.add_argument(
        "--fold-dirs",
        type=str,
        nargs="+",
        default=None,
        help="Only consider structures under these AF3 output directory names "
        "(e.g. output_fold_all output_fold_all2). Ranking is over the filtered set.",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        default="specificities",
        help="Subdirectory of the job path for specificity outputs.",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=30,
        help="NA-MPNN forward passes per structure (--number_of_batches).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="NA-MPNN sampling temperature (--temperature).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="GPUs per task. -1 (auto) requests 1 on gpu/preempted; pass 0 to run "
        "CPU-only on a partition's CPU nodes (e.g. --partition preempted --gpus 0).",
    )
    parser.add_argument(
        "--design-na-only",
        action="store_true",
        help="Fix protein chains so NA-MPNN conditions DNA predictions on the "
        "true amino acid identities (default: protein and DNA both masked).",
    )
    parser.add_argument(
        "--proteins-file",
        type=str,
        default=None,
        help="Newline-delimited file of DBIDs to restrict the run to.",
    )
    args = parser.parse_args()

    jobdb_path = Path(args.jobdb)
    jobdb_dir = jobdb_path.parent

    proteins = get_proteins(args.dataset, jobdb_path, proteins_file=args.proteins_file)
    structs = get_all_structs(
        proteins, jobdb_path, n_structs=args.n_structs, fold_dirs=args.fold_dirs
    )

    spec_dir = jobdb_dir / args.spec_dir
    structs = [
        (protein, motif, dna, seed)
        for protein, motif, dna, seed in structs
        if not (
            spec_dir / protein / "specificity" / f"{protein}_{motif}_{dna}_{seed}.npz"
        ).exists()
    ]
    print(f"{len(structs)} structures to run ({len(proteins)} proteins).")

    if not structs:
        print("Nothing to submit; all outputs already exist.")
        return

    logs_dir = jobdb_dir / "slurm_logs"
    logs_dir.mkdir(exist_ok=True)

    # Tag per-submission files by spec-dir in case of sweep/f
    tag = args.spec_dir.replace("/", "_")
    job_list = write_job_list(structs, jobdb_dir, tag=tag)
    n_jobs = len(structs)
    slurm_script = generate_slurm_script(
        job_list,
        n_jobs,
        jobdb_dir,
        args.dataset,
        args.partition,
        args.spec_dir,
        n_batches=args.n_batches,
        softmax_temp=args.temperature,
        gpus=args.gpus,
        design_na_only=args.design_na_only,
    )

    script_file = jobdb_dir / f"run_specificity_{tag}.sh"
    with open(script_file, "w") as f:
        f.write(slurm_script)

    result = sp.run(
        ["sbatch", "--parsable", str(script_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    print(
        f"Submitted SLURM array job with ID {result.stdout.strip()}"
        f" ({len(structs)} jobs)"
    )


if __name__ == "__main__":
    main()
