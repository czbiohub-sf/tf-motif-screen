"""
Extract motif hits, denovo and from database, from HOMER results.

Ben Iovino  8/19/25    Biohub
"""

import subprocess as sp
from pathlib import Path

import pandas as pd

from af3tfm.motif import parse_pwm_file


def get_denovo_results(results_path: Path, hits_file_path: Path):
    """
    Concatenates all non-redundant motif files from HOMER denovo results into a single file.

    Args:
        results_path : Path
            Path to the directory containing HOMER results subdirectories.
        hits_file_path : Path
            Path to the output file where concatenated motifs will be saved.
    """

    sp.run(["touch", hits_file_path])

    if hits_file_path.stat().st_size != 0:
        print(
            f"{hits_file_path} already exists and is not empty. Please remove or rename it before running this script."
        )
        return

    with open(hits_file_path, "a") as motifs_file:
        for path in results_path.iterdir():
            if path.is_dir():
                if path.name == "slurm_logs":
                    continue
                nr_motif_file_path = f"{path}/nonRedundant.motifs"
                sp.run(["cat", nr_motif_file_path], stdout=motifs_file, check=True)


def get_known_results(results_path: Path) -> set[str]:
    """
    Returns all motif names from HOMER knownResults.txt files with P-value < 1e-12.

    Args:
        results_path : Path
            Path to the directory containing HOMER results subdirectories.

    Returns:
        set[str]: Set of motif names with significant hits.
    """

    results = set()
    for results_dir in results_path.iterdir():
        if results_dir.is_dir():
            if results_dir.name == "slurm_logs":
                continue
            results_file_path = f"{results_dir}/knownResults.txt"

            # Get top motifs from results
            results_df = pd.read_csv(results_file_path, sep="\t", header=0)
            results_df = results_df[results_df["P-value"] < 1e-12]

            # Add motif names to the set
            for row in results_df.iterrows():
                results.add(row[1]["Motif Name"])

    return results


def assign_source_clusters(results_path: Path, clustered_path: Path, out_path: Path):
    """
    Assign each representative motif in jaspar_clustered/nonRedundant.motifs back
    to the UMAP peak cluster (0-35) it originated from.

    Args:
        results_path : Path
            Path to the directory containing per-cluster HOMER results (0-35).
        clustered_path : Path
            Path to the jaspar_clustered output directory.
        out_path : Path
            Path to write the output TSV.
    """

    # Build mapping: source name -> list of source clusters
    name_to_clusters: dict[str, list[int]] = {}
    for cluster_dir in sorted(results_path.iterdir(), key=lambda p: int(p.name)):
        cluster_id = int(cluster_dir.name)

        denovo_file = cluster_dir / "nonRedundant.motifs"
        if denovo_file.exists():
            for name in _motif_names(denovo_file):
                name_to_clusters.setdefault(name, []).append(cluster_id)

        known_file = cluster_dir / "knownResults.txt"
        if known_file.exists():
            for name in _known_names(known_file):
                name_to_clusters.setdefault(name, []).append(cluster_id)

    # Match each representative to its source cluster(s)
    reps = _motif_names(clustered_path / "nonRedundant.motifs")
    rows = []
    for rep in reps:
        consensus = rep.split(",")[0]
        source_clusters = []
        for source_name, clusters in name_to_clusters.items():
            if rep.startswith(source_name):
                source_clusters = clusters
                break
        rows.append(
            {
                "consensus": consensus,
                "motif": rep,
                "source_cluster": source_clusters[0] if source_clusters else None,
                "source_clusters": source_clusters,
            }
        )

    df = pd.DataFrame(rows)

    missing = df[df["source_cluster"].isna()]
    ambiguous = df[df["source_clusters"].map(len) > 1]
    if not missing.empty:
        print(f"WARNING: {len(missing)} representatives not found in any cluster.")
    if not ambiguous.empty:
        print(
            f"NOTE: {len(ambiguous)} representatives found in multiple clusters (first used):"
        )
        for _, row in ambiguous.iterrows():
            print(f"  {row['consensus']} -> clusters {row['source_clusters']}")

    df[["consensus", "motif", "source_cluster", "source_clusters"]].to_csv(
        out_path, sep="\t", index=False
    )


def _motif_names(path: Path) -> list[str]:
    """
    Return name fields (second tab field) from a HOMER .motif file.
    """

    names = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                names.append(line[1:].rstrip().split("\t")[1])
    return names


def _known_names(path: Path, pval_thresh: float = 1e-12) -> list[str]:
    """
    Return motif names from knownResults.txt passing p-value threshold.
    """

    df = pd.read_csv(path, sep="\t", header=0)
    return df[df["P-value"] < pval_thresh]["Motif Name"].tolist()


def main():
    """ """

    results_path = Path("data/tools/homer/jaspar/output")
    denovo_hits_file_path = Path("data/tools/homer/hits.NonRedundant.motif")
    jaspar_hits_file_path = Path("data/tools/homer/hits.jaspar.motif")
    all_hits_file_path = Path("data/tools/homer/hits.all.motif")
    clustered_path = Path("data/tools/homer/jaspar_clustered")

    get_denovo_results(results_path, denovo_hits_file_path)
    known = get_known_results(results_path)
    pwms = parse_pwm_file("data/tools/homer/jaspar.homer.motif")

    # Check which known motifs were found and write to file
    with open(jaspar_hits_file_path, "w", encoding="utf-8") as out_file:
        for name in known:
            if name in pwms:
                out_file.write(f">{pwms[name][0]}\t{name}\t6.0\t0\t0\n")
                for row in pwms[name][1]:
                    formatted_row = "\t".join(f"{p:.6f}" for p in row)
                    out_file.write(f"{formatted_row}\n")
            else:
                print(f"Warning: {name} not found in PWM file.")

    # Combine all hits and run HOMER clustering
    sp.run(
        ["cat", denovo_hits_file_path, jaspar_hits_file_path],
        stdout=open(all_hits_file_path, "w"),
        check=True,
    )
    cmd = f"compareMotifs.pl {all_hits_file_path} {clustered_path}"
    sp.run(cmd, shell=True, check=True)

    assign_source_clusters(
        results_path,
        clustered_path,
        Path("data/tools/homer/motifs.tsv"),
    )


if __name__ == "__main__":
    main()
