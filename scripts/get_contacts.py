"""
Scalable, sharded protein-DNA interface extraction for large TF screens.

Submit as a single SLURM job (resource specs live here, not in a .slurm file):
  python scripts/get_contacts.py submit --run hero --dataset tfdb4_rep --shards 100

Or run the work directly (this is what the submitted job executes):
  python scripts/get_contacts.py run --run hero --dataset tfdb4_rep --shards 100

Outputs (under data/jobs/<run>/):
  interface_shards/shard_NNNN.parquet - per-shard aggregated res_df
  interface_residues.parquet          - final concatenated table

Ben Iovino 8/4/2026 Biohub
"""

import argparse
import sqlite3
import subprocess as sp
from pathlib import Path

import pandas as pd
from af3tfm.config import CONFIG
from af3tfm.contacts import balance_shards, run_shard
from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT


def _job_dir(run: str) -> Path:
    """
    Base directory holding a run's jobdb and interface outputs.

    Args:
        run: job run name under data/jobs/

    Returns:
        Path to data/jobs/<run>/
    """
    return Path(f"{DATA_DIR}/jobs/{run}")


def _shard_path(run: str, i: int) -> Path:
    """
    Location of one shard's aggregated parquet (zero-padded for stable sort).

    Args:
        run: job run name under data/jobs/
        i: shard index

    Returns:
        Path to data/jobs/<run>/interface_shards/shard_NNNN.parquet
    """
    return _job_dir(run) / "interface_shards" / f"shard_{i:04d}.parquet"


def _final_path(run: str) -> Path:
    """
    Location of the final concatenated per-(TF, residue) table.

    Args:
        run: job run name under data/jobs/

    Returns:
        Path to data/jobs/<run>/interface_residues.parquet
    """
    return _job_dir(run) / "interface_residues.parquet"


def _connect_ro(run: str) -> sqlite3.Connection:
    """
    Read-only jobdb connection (safe to open while other readers run).

    Args:
        run: job run name under data/jobs/

    Returns:
        read-only sqlite3 connection to data/jobs/<run>/jobdb.sqlite
    """
    db = _job_dir(run) / "jobdb.sqlite"
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _tf_sizes(run: str) -> dict:
    """
    Structure count per TF, straight from the jobdb.

    Args:
        run: job run name under data/jobs/

    Returns:
        dict mapping protein (dbid) -> number of structures
    """
    conn = _connect_ro(run)
    try:
        rows = conn.execute(
            "SELECT protein, COUNT(*) FROM runs GROUP BY protein"
        ).fetchall()
    finally:
        conn.close()
    return {p: n for p, n in rows}


def _tasks_for(run: str, tfs: list[str]) -> list[tuple]:
    """
    Per-structure work items for the given TFs.

    Args:
        run: job run name under data/jobs/
        tfs: TFs (dbids) to pull structures for

    Returns:
        list of (structure_path, dbid, is_non) tuples; is_non flags the
        negative-control ("-non") motif for each structure
    """
    conn = _connect_ro(run)
    qmarks = ",".join("?" * len(tfs))
    try:
        rows = conn.execute(
            f"SELECT structure_path, protein, motif FROM runs "
            f"WHERE protein IN ({qmarks})",
            list(tfs),
        ).fetchall()
    finally:
        conn.close()
    return [(sp_, p, m.endswith("-non")) for sp_, p, m in rows]


def _build_shards(run: str, n_shards: int) -> list[list[str]]:
    """
    Deterministic shard -> TF assignment straight from the jobdb.

    Args:
        run: job run name under data/jobs/
        n_shards: number of shards to split the TFs into

    Returns:
        list of shards, each a list of TFs (dbids) assigned to that shard
    """
    return balance_shards(_tf_sizes(run), n_shards)


def _name_map(dataset: str) -> dict:
    """
    DBID (upper) -> TF_Name from from full dataset, overlaid with the run's own data.

    Args:
        dataset: dataset name (tfdb4 or tfdb4_rep)

    Returns:
        dict mapping DBID (upper) -> TF_Name, for readability in the final table.
    """

    def load(ds):
        t = pd.read_csv(DS[ds].seq_csv)
        return {str(d).upper(): n for n, d in zip(t["TF_Name"], t["DBID"])}

    m = {}
    if "tfdb4" in DS:
        m.update(load("tfdb4"))  # broad coverage
    m.update(load(dataset))  # run's dataset takes precedence
    return m


def _concat(run: str, dataset: str, n_shards: int) -> None:
    """
    Merge every shard parquet into the final per-(TF, residue) table.

    Args:
        run: job run name under data/jobs/
        dataset: dataset name (tfdb4 or tfdb4_rep)
        n_shards: number of shards to merge
    """
    shards = _build_shards(run, n_shards)
    parts, missing = [], []
    for i in range(len(shards)):
        sp_path = _shard_path(run, i)
        if sp_path.exists():
            parts.append(pd.read_parquet(sp_path))
        else:
            missing.append(i)

    def _fmt(ids):  # keep the list readable even if many shards failed
        return str(ids) if len(ids) <= 20 else f"{ids[:20]} ... (+{len(ids) - 20} more)"

    if missing:
        print(f"WARNING: {len(missing)} shard(s) missing: {_fmt(missing)}")
    res_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # TF name for readability (full-set fallback -> ~99.8% coverage)
    if not res_df.empty:
        res_df["tf"] = res_df["dbid"].str.upper().map(_name_map(dataset))
        n_unmapped = res_df["tf"].isna().sum()
        if n_unmapped:
            print(
                f"note: {n_unmapped} rows "
                f"({res_df.loc[res_df['tf'].isna(), 'dbid'].nunique()} TFs) "
                f"have no TF_Name in any table"
            )

    out = _final_path(run)
    res_df.to_parquet(out, index=False)  # parquet only; avoids clobbering the
    # notebook's interface_residues.csv (same schema, use .to_csv() to export)
    n_tf = res_df["dbid"].nunique() if not res_df.empty else 0
    print(f"Wrote {out} ({len(res_df)} TF-residue rows, {n_tf} TFs)")
    if missing:
        print(
            f"NOTE: re-run to fill missing shards, then concat re-runs: {_fmt(missing)}"
        )


def cmd_run(args):
    """Process every shard in order, then concatenate. Resumable."""
    shards = _build_shards(args.run, args.shards)
    print(f"{len(shards)} shards, sequential")
    for i, tfs in enumerate(shards):
        sp_path = _shard_path(args.run, i)
        if sp_path.exists():
            print(f"[{i + 1}/{len(shards)}] shard exists, skipping")
            continue
        tasks = _tasks_for(args.run, tfs)
        print(f"[{i + 1}/{len(shards)}] {len(tfs)} TFs, {len(tasks)} structures")
        run_shard(
            tasks,
            sp_path,
            max_workers=args.workers,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
        )
    _concat(args.run, args.dataset, args.shards)


def _slurm_script(args) -> str:
    """
    SLURM batch script for one sequential `run` job (specs baked in here).

    Args:
        args: parsed CLI args (uses run, dataset, shards, workers, chunk_size,
            timeout)

    Returns:
        the #SBATCH script body, which re-invokes this script's `run` command
    """
    log_dir = _job_dir(args.run) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mail_user = (
        f"#SBATCH --mail-user={CONFIG['mail_user']}\n" if CONFIG["mail_user"] else ""
    )
    return f"""#!/bin/bash
#SBATCH --job-name=iface_{args.run}
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.workers}
#SBATCH --mail-type=FAIL
{mail_user}#SBATCH --output={log_dir}/iface_%j.out
#SBATCH --error={log_dir}/iface_%j.err

module load anaconda
module load data.science
conda activate af3-tfm

python {PROJECT_ROOT}/scripts/get_contacts.py run \\
    --run {args.run} --dataset {args.dataset} --shards {args.shards} \\
    --workers {args.workers} --chunk-size {args.chunk_size} --timeout {args.timeout}
"""


def cmd_submit(args):
    """Write the batch script and submit it as a single SLURM job."""
    script = _slurm_script(args)
    script_file = _job_dir(args.run) / "interface_contacts.sh"
    script_file.write_text(script)
    res = sp.run(
        ["sbatch", "--parsable", str(script_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = res.stdout.strip()
    print(f"Submitted job {job_id} ({args.shards} shards)")
    print(f"-> {_final_path(args.run)}")
    print(f"logs: {_job_dir(args.run)}/logs/iface_{job_id}.out")


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, help="job run name under data/jobs/")
    common.add_argument("--dataset", required=True, choices=list(DS))
    common.add_argument(
        "--shards",
        type=int,
        default=100,
        help="number of shards (default 100); memory/resume "
        "granularity, higher = smaller peak memory",
    )

    worker = argparse.ArgumentParser(add_help=False)
    worker.add_argument(
        "--workers",
        type=int,
        default=16,
        help="parallel workers per shard (also SLURM cpus-per-task)",
    )
    worker.add_argument("--chunk-size", type=int, default=2000)
    worker.add_argument("--timeout", type=float, default=60)

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser(
        "submit", parents=[common, worker], help="submit as a single SLURM job"
    )
    p_submit.set_defaults(func=cmd_submit)

    p_run = sub.add_parser(
        "run",
        parents=[common, worker],
        help="process all shards sequentially, then concat",
    )
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
