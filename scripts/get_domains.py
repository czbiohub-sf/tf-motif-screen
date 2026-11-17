"""
Get protein domains using interproscan. Determine if the domains are DNA-binding via the paired
aligned error values between protein and DNA sequences from AlphaFold3 predictions.

Ben Iovino  2/28/25    CZ-Biohub
"""

import json
import os
import re
import subprocess as sp
from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from af3tfm.dataset import DATA_DIR, DS


def _discover_run_worker(direc_path: str):
    """
    Worker for determine_dna_binding — inspects one AF3 job directory and returns
    the protein it belongs to plus the paths needed for DNA-binding analysis.

    Args:
        direc_path (str): Path to a single job directory (child of a fold dir).

    Returns:
        (tf_upper, conf_path, iptm) where
            tf_upper  - uppercase protein DBID parsed from the directory name
            conf_path - path to the *_confidences.json file
            iptm      - ipTM from the summary file (0.0 if unavailable)
        or None if the directory doesn't hold a usable prediction.
    """
    try:
        direc = Path(direc_path)

        # Parse TF and motif names (same pattern as jobdb.process_runs)
        pattern = r"^(?P<protein>[a-z0-9]+(?:_\d+)?)_(?P<motif>.+?)_(?:top_)?(?P<seq_num>\d+)$"
        match = re.match(pattern, direc.name)
        if not match:
            return None
        tf_upper = match.group("protein").upper()

        # Find valid output (seed) directory holding a non-empty model.cif
        output_path = None
        for d in direc.iterdir():
            if d.is_dir() and any(d.iterdir()):
                for file in d.iterdir():
                    if str(file).endswith("model.cif") and file.stat().st_size > 0:
                        output_path = d
                        break
            if output_path:
                break
        if output_path is None:
            return None

        # Locate the confidences and summary files
        conf_path, summary_path = None, None
        for file in output_path.iterdir():
            name = str(file)
            if name.endswith("_summary_confidences.json"):
                summary_path = file
            elif name.endswith("_confidences.json"):
                conf_path = file
        if conf_path is None:
            return None

        iptm = 0.0
        if summary_path is not None:
            try:
                with open(summary_path) as f:
                    iptm = float(json.load(f).get("iptm", 0.0) or 0.0)
            except (FileNotFoundError, json.JSONDecodeError):
                iptm = 0.0

        return (tf_upper, str(conf_path), iptm)
    except Exception:
        return None


def _determine_dbd_worker(args: tuple):
    try:
        return _determine_dbd_worker_inner(args)
    except Exception:
        return None


def _determine_dbd_worker_inner(args: tuple):
    """
    Worker for determine_dna_binding — reads PAE from the confidences file
    for each AF3 run of one TF and flags its DNA-binding domain(s).

    Two modes
    - multidom=True: (tandem-array families, e.g. zf-C2H2 / ZBTB): returns
      every domain whose per-run occupancy clears min_occupancy >= 0.5
    - multidom=False (all other families): returns the single domain with the
      most protein-DNA contacts (argmax of the aggregated contact score)

    Args:
        args: (tf_upper, conf_paths, domains, min_occupancy, multidom) where
              tf_upper      - uppercase DBID (e.g. "ENSDARG00000005382")
              conf_paths    - list of paths to *_confidences.json files for this TF
              domains       - list of (start, end) tuples for all domains
              min_occupancy - fraction of runs a domain must be in-region to pass
              multidom      - allow multiple DBDs via the occupancy rule if True

    Returns:
        (tf_upper, [(start, end), ...]) passing domains, or None if no contacts.
    """
    tf_upper, conf_paths, domains, min_occupancy, multidom = args

    # Per-domain: number of runs the domain overlapped a high-contact region.
    # Also pool region membership across runs for the tie-break score.
    n_domains = len(domains)
    occ_counts = np.zeros(n_domains, dtype=float)
    n_runs = 0
    all_regions = []
    for conf_path in conf_paths:
        try:
            with open(conf_path) as f:
                conf = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        pae = np.array(conf["pae"])
        token_res = conf["token_chain_ids"]

        # Identify DNA positions and get PAE for protein-DNA contacts
        dna_pos = [i for i, x in enumerate(token_res) if x in ("B", "C")]
        b_pae = pae[dna_pos[0] : dna_pos[-1] + 1, : dna_pos[0]]

        contact_counts = (b_pae < 5).sum(axis=0)  # Count contacts per protein position
        nonzero = contact_counts[
            contact_counts > 0
        ]  # Consider only positions with contacts for thresholding
        if len(nonzero) == 0:
            continue
        threshold = nonzero.mean()  # Dynamic threshold: mean of non-zero contact counts

        # Build boolean in-region mask for this run (contiguous above-threshold
        # stretches, closed at a zero-contact position)
        inregion = np.zeros(len(contact_counts), dtype=bool)
        regions = []
        region = []
        for i, count in enumerate(contact_counts):
            if count >= threshold:
                region.append(i)
            if count == 0 and region:
                regions.append(region)
                region = []
        if region:
            regions.append(region)
        if not regions:
            continue
        for region in regions:
            for pos in region:
                inregion[pos] = True

        # Count this run toward each domain that overlaps a region (1-based coords)
        n_runs += 1
        all_regions.append(regions)
        for di, (start, end) in enumerate(domains):
            if inregion[start - 1 : end].any():
                occ_counts[di] += 1

    if n_runs == 0:
        return None

    occupancy = occ_counts / n_runs

    # Aggregated contact score per domain (mean contact frequency across its
    # residues, pooled over runs): the single-domain metric, and the tie-break
    # for the multi-domain floor.
    flat = [pos for run in all_regions for region in run for pos in region]
    max_end = max(dom[1] for dom in domains)
    counts = np.bincount(flat, minlength=max_end)
    scores = np.array([counts[s - 1 : e].mean() for s, e in domains])

    if multidom:
        # Pass every domain clearing the occupancy threshold; floor to the single best domain
        passing = [di for di in range(n_domains) if occupancy[di] >= min_occupancy]
        if not passing:
            passing = [
                max(
                    range(n_domains),
                    key=lambda di: (occupancy[di], scores[di], -domains[di][0]),
                )
            ]
    else:
        # Single DBD: the domain with the most protein-DNA contacts
        passing = [int(np.argmax(scores))]

    return (tf_upper, [(int(domains[di][0]), int(domains[di][1])) for di in passing])


def make_fasta(df_path: str, fasta_file_path: str):
    """
    Writes all fasta sequences in the dataframe to one file.

    Args:
        df_path (str):
            path to the dataframe.
        fasta_file_path (str):
            path to the fasta file.
    """

    tf_df = pd.read_csv(df_path)

    # Get the fasta sequences and write to file
    seqs = {}
    for row in tf_df.itertuples():
        seqs[row.DBID] = row.Sequence
    with open(fasta_file_path, "w") as f:
        for dbid, seq in seqs.items():
            f.write(f">{dbid}\n{seq}\n")


def generate_ipscan_script(fasta_file_path: str, results_file_path: str) -> str:
    """
    Generates a slurm script to run interproscan.

    Args:
        fasta_file_path (str):
            path to the fasta file.
        results_file_path (str):
            path to the results file.

    Returns:
        slurm_script (str):
    """

    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=ipscan
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=data/tools/ipscan/%x_%A_%a.out
#SBATCH --error=data/tools/ipscan/%x_%A_%a.err
#SBATCH --mail-type=FAIL

module load oracle-java/jdk-17.0.1

{DATA_DIR}/tools/ipscan/interproscan/interproscan.sh -dp -i {fasta_file_path} -o {results_file_path} -f JSON -cpu 32 -appl Pfam,ProSiteProfiles
"""

    return slurm_script


def parse_results(results_file_path: str, doms_df_file_path: str):
    """
    Parses the results file and writes the domains to a csv file.

    Args:
        results_file_path (str):
            path to the results file.
        doms_df_file_path (str):
            path to save the dataframe with domain information.
    """

    with open(results_file_path) as f:
        results = json.load(f)["results"]

    domains = []
    databases = ["PFAM", "PROSITE_PROFILES"]
    for result in results:
        # InterProScan groups identical sequences under one result. Every
        # gene in xref needs the same domain calls, not just the first one.
        dbids = [xref["name"] for xref in result["xref"]]
        for match in result["matches"]:
            signature = match["signature"]
            if signature["signatureLibraryRelease"]["library"] not in databases:
                continue

            # Prefer the InterPro entry's accession/description. Fall back to the
            # signature's own when the entry is None
            entry = signature["entry"]
            if entry is not None:
                accession = entry["accession"]
                description = entry["description"]
            else:
                accession = signature["accession"]
                description = signature.get("description") or signature.get("name")

            for location in match[
                "locations"
            ]:  # There can be multiple locations for a single domain
                for dbid in dbids:
                    domains.append(
                        {
                            "DBID": dbid,
                            "library": signature["signatureLibraryRelease"]["library"],
                            "accession": accession,
                            "description": description,
                            "start": location["start"],
                            "end": location["end"],
                        }
                    )

    df = pd.DataFrame(domains).set_index("DBID")
    df.to_csv(doms_df_file_path)


def collapse_domains(tf_df_file_path: str, doms_df_file_path: str):
    """
    Collapse similar domains into one.

    Args:
        tf_df_file_path (str):
            path to the dataframe with protein information.
        doms_df_file_path (str):
            path to the dataframe.
    """

    doms_df = pd.read_csv(doms_df_file_path)
    tf_df = pd.read_csv(tf_df_file_path)

    # For each DBID, get domains
    collapsed_domains = []
    for dbid in doms_df["DBID"].unique():
        dbid_df = doms_df[doms_df["DBID"] == dbid]

        # Get info for each domain
        doms = dbid_df[["accession", "start", "end", "description"]].values
        blacklist = []
        for i, dom1 in enumerate(doms):
            dom1 = list(dom1)
            if i in blacklist:  # Skip domains that have already been merged
                continue
            for j, dom2 in enumerate(doms):
                if i >= j:  # Only compare unique pairs
                    continue
                if j in blacklist:  # Skip domains that have already been merged
                    continue

                # Merge domain1 and domain2 if they overlap
                if abs(dom1[1] - dom2[1]) < 20 or abs(dom1[2] - dom2[2]) < 20:
                    start = min(dom1[1], dom2[1])
                    end = max(dom1[2], dom2[2])
                    description = dom1[3]
                    dom1 = [dom1[0], start, end, description]
                    blacklist.append(j)

            # Add to list for easy df creation
            collapsed_domains.append(
                {
                    "DBID": dbid,
                    "TF_Name": None,
                    "accession": dom1[0],
                    "description": dom1[3],
                    "start": dom1[1],
                    "end": dom1[2],
                }
            )

    df = pd.DataFrame(collapsed_domains).set_index("DBID")
    mapping = tf_df.set_index("DBID")["TF_Name"].to_dict()
    df["TF_Name"] = df.index.map(mapping)
    df.to_csv(doms_df_file_path)


def determine_dna_binding(
    fold_dirs: list[Path],
    doms_df: pd.DataFrame,
    tf_family: dict[str, str] = None,
    multidom_families: set[str] = frozenset({"zf-C2H2", "ZBTB"}),
    max_workers: int = None,
    min_occupancy: float = 0.5,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Determine DNA-binding domains in parallel. Walks the given AF3 output
    directories directly, groups the confidences files by TF, keeps each
    TF's top-ipTM runs, and marks DNA-binding domains via each TF's protein-DNA PAE.

    Args:
        fold_dirs (list[Path]):
            AF3 output directories to walk (e.g. data/jobs/hero/output_fold_all).
            Each child directory is one prediction job.
        doms_df (pd.DataFrame):
            DataFrame containing domain information.
        tf_family (dict[str, str]):
            Map of uppercase DBID -> family name. TFs absent from the map (or when
            the map is None) are treated as single-DBD.
        multidom_families (set[str]):
            Family names allowed multiple DBDs via the occupancy rule.
        max_workers (int):
            Number of worker processes (None = CPU count).
        min_occupancy (float):
            Fraction of a TF's runs a domain must overlap a high-contact region
            to be flagged DNA-binding, for multi-DBD families (default 0.5).
        quantile (float):
            Per-TF ipTM quantile; only runs at or above it are analyzed (0.95
            keeps the top ~5% of each TF's runs, matching the old DB behavior).

    Returns:
        pd.DataFrame:
            Updated DataFrame with a "DNA-Binding" column indicating DNA-binding domains.
    """

    tf_family = tf_family or {}

    # Build per-TF domain lookup: uppercase DBID -> list of (start, end)
    tf_to_domains: dict[str, list[tuple[int, int]]] = {
        dbid: list(group[["start", "end"]].itertuples(index=False, name=None))
        for dbid, group in doms_df.groupby("DBID")
    }

    # Walk every fold dir's job directories and collect (tf, conf_path, ipTM)
    print(f"Reading {len(fold_dirs)} fold director(ies):")
    job_dirs = []
    for fold_dir in fold_dirs:
        fold_dir = Path(fold_dir)
        n_before = len(job_dirs)
        job_dirs.extend(str(d) for d in fold_dir.iterdir() if d.is_dir())
        print(f"  {fold_dir}: {len(job_dirs) - n_before} job directories")
    print(f"Total: {len(job_dirs)} job directories to process")

    records: list[tuple[str, str, float]] = []
    with Pool(processes=max_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_discover_run_worker, job_dirs, chunksize=10),
            total=len(job_dirs),
            desc="Discovering runs",
        ):
            if result is not None:
                records.append(result)

    # Group confidences paths per TF (only TFs we have domains for)
    per_tf: dict[str, list[tuple[str, float]]] = {}
    for tf_upper, conf_path, iptm in records:
        if tf_upper not in tf_to_domains:
            continue
        per_tf.setdefault(tf_upper, []).append((conf_path, iptm))

    # Keep each TF's top-ipTM runs (>= per-TF quantile)
    tf_to_conf_paths: dict[str, list[str]] = {}
    for tf_upper, recs in per_tf.items():
        iptms = np.array([it for _, it in recs])
        threshold = np.quantile(iptms, quantile)
        conf_paths = [cp for cp, it in recs if it >= threshold]
        if conf_paths:
            tf_to_conf_paths[tf_upper] = conf_paths

    args_list = [
        (
            tf_upper,
            conf_paths,
            tf_to_domains[tf_upper],
            min_occupancy,
            tf_family.get(tf_upper) in multidom_families,
        )
        for tf_upper, conf_paths in tf_to_conf_paths.items()
    ]

    doms_df["DNA-Binding"] = False
    with Pool(processes=max_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_determine_dbd_worker, args_list, chunksize=1),
            total=len(args_list),
            desc="Determining DNA-binding domains",
        ):
            if result is None:
                continue
            tf_upper, passing = result
            for start, end in passing:
                mask = (
                    (doms_df["DBID"] == tf_upper)
                    & (doms_df["start"] == start)
                    & (doms_df["end"] == end)
                )
                doms_df.loc[mask, "DNA-Binding"] = True

    return doms_df


def _str2bool(value: str) -> bool:
    """Parse a boolean argparse value (True/False, yes/no, 1/0)."""
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _positive_int(value: str) -> int:
    """Parse a positive CPU/worker limit."""
    number = int(value)
    if number < 1:
        raise ArgumentTypeError("Worker count must be positive")
    return number


def _available_workers() -> int:
    """Respect CPU affinity when running inside an existing allocation."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    """ """

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument(
        "--ipscan",
        type=_str2bool,
        required=True,
        help="If True, submit the InterProScan SLURM job to (re)generate the "
        "results file; if False, skip it and read the existing results file on disk.",
    )
    parser.add_argument(
        "--fold-dirs",
        type=str,
        nargs="+",
        required=True,
        help="AF3 output directories to walk for confidences files "
        "(e.g. data/jobs/hero/output_fold_all data/jobs/hero/output_fold_all2). "
        "Each child directory is one prediction job.",
    )
    parser.add_argument(
        "--min_occupancy",
        type=float,
        default=0.5,  # Default set to allow 2-3 zf-C2H2 domains to pass
        help="Fraction of a TF's runs a domain must contact DNA to be flagged DNA-binding",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=None,
        help="Parallel workers for AF3 discovery and DBD analysis (default: allowed CPUs).",
    )
    parser.add_argument(
        "--ipscan-results",
        type=Path,
        help="Read/write this InterProScan JSON instead of the dataset's default path.",
    )
    parser.add_argument(
        "--domains-output",
        type=Path,
        help="Write domains to this CSV instead of replacing the dataset's domain table.",
    )
    args = parser.parse_args()

    dataset = DS[args.dataset]
    Path(f"{DATA_DIR}/tools/ipscan/{dataset.name}").mkdir(parents=True, exist_ok=True)
    fasta_file_path = (
        f"{DATA_DIR}/tools/ipscan/{dataset.name}/{dataset.name}_seqs.fasta"
    )
    results_file_path = args.ipscan_results or (
        f"{DATA_DIR}/tools/ipscan/{dataset.name}/{dataset.name}_results.json"
    )
    doms_file_path = args.domains_output or dataset.dom_csv

    # Get interproscan domain predictions
    if args.ipscan:
        make_fasta(dataset.seq_csv, fasta_file_path)
        slurm_script = generate_ipscan_script(fasta_file_path, results_file_path)
        sp.run(["sbatch", "--wait"], input=slurm_script, text=True, check=True)
    parse_results(results_file_path, doms_file_path)
    collapse_domains(dataset.seq_csv, doms_file_path)

    # Determine if they are DNA-binding
    doms_df = pd.read_csv(doms_file_path)
    tf_df = pd.read_csv(dataset.seq_csv)
    tf_family = {
        str(dbid).upper(): fam for dbid, fam in zip(tf_df["DBID"], tf_df["Family"])
    }
    doms_df = determine_dna_binding(
        [Path(d) for d in args.fold_dirs],
        doms_df,
        tf_family=tf_family,
        max_workers=args.max_workers or _available_workers(),
        min_occupancy=args.min_occupancy,
    )
    doms_df.to_csv(doms_file_path, index=False)


if __name__ == "__main__":
    main()
