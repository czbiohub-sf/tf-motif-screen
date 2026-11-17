"""
Here we are looking at the per-family clusters returned by mmseqs2 on the TFDB dataset. If the
similarity between sequences in a cluster is relatively high, while the similarity between
clusters is relatively lower, then this suggests that the clusters may represent different
domain architectures within the same family.
"""

# %%
import subprocess as sp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import AlignIO
from Bio.Align import substitution_matrices

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()


def parse_fasta_clusters(file_path: str) -> list[dict[str, str]]:
    """
    Returns a list of dictionaries where each dictionary represents a cluster of sequences.

    Args:
        file_path (str): Path to the FASTA file containing clustered sequences.

    Returns:
        list[dict[str, str]]: ID is sequence identifier, value is sequence string.
    """

    with open(file_path, "r") as f:
        # Split by '>' but keep the character to identify headers
        content = f.read().strip().split(">")

    clusters, current_cluster = [], {}
    for block in content:
        if not block:
            continue

        lines = block.strip().split("\n")
        header = lines[0].split()[0]  # Get ID only, ignore metadata
        seq = "".join(lines[1:])  # Join multi-line sequences

        if not seq:  # If there is no sequence, this is a cluster rep line.
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = {}
        else:
            current_cluster[header] = seq

    if current_cluster:  # Append the final cluster
        clusters.append(current_cluster)

    return clusters


def calculate_similarity(align1: str, align2: str) -> float:
    """
    Returns the percent similarity between two aligned sequences.
    Similarity is defined as (similar pairs) / (total non-gap pairs).

    Args:
        align1 (str): First aligned sequence.
        align2 (str): Second aligned sequence.

    Returns:
        float: Percent similarity between the two sequences.
    """

    matrix = substitution_matrices.load("BLOSUM62")

    positives = 0
    total_comparisons = 0
    for a, b in zip(align1, align2):
        if a == "-" and b == "-":  # Skip gaps in both sequences
            continue

        total_comparisons += 1
        if a == "-" or b == "-":  # Penalize gaps
            continue
        if (a, b) in matrix:  # Substitution score
            pair_score = matrix[a, b]
        else:
            pair_score = matrix[b, a]

        if pair_score > 0:  # Positive score is similar
            positives += 1

    sim = (positives / total_comparisons) * 100
    return sim


def get_similarity_df(alignment_file: str) -> pd.DataFrame:
    """
    Returns a DataFrame representing the percent similarity between sequences in an alignment file.

    Args:
        alignment_file (str): Path to the alignment file in FASTA format.

    Returns:
        pd.DataFrame: DataFrame with sequence IDs as both index and columns, containing percent similarity values.
    """

    alignment = AlignIO.read(alignment_file, "fasta")
    num_seqs = len(alignment)
    similarity_matrix = np.zeros((num_seqs, num_seqs))

    for m in range(num_seqs):
        for n in range(num_seqs):
            if m == n:
                similarity_matrix[m, n] = 100.0
            elif m < n:  # Avoid redundant calculations
                sim = calculate_similarity(str(alignment[m].seq), str(alignment[n].seq))
                similarity_matrix[m, n] = similarity_matrix[n, m] = sim

    seq_ids = [record.id for record in alignment]
    similarity_df = pd.DataFrame(similarity_matrix, index=seq_ids, columns=seq_ids)

    # Calculate mean similarity outside diagonal
    sim_values = similarity_df.values
    mean_sim = (sim_values.sum() - np.trace(sim_values)) / (sim_values.size - num_seqs)

    return similarity_df, mean_sim


def run_mafft(
    data: dict[str, str],
    input_file: str = "clusters.fasta",
    output_file: str = "clusters.mfa",
):
    """
    Writes sequences to a FASTA file and runs MAFFT to align them.

    Args:
        data (dict[str, str]): Dictionary with sequence IDs as keys and sequences as values.
        input_file (str): Path to the input FASTA file for MAFFT.
        output_file (str): Path to the output alignment file from MAFFT.
    """

    mafft_cmd = [
        "module",
        "load",
        "mafft",
        "&&",
        "mafft",
        "--quiet",
        "--auto",
        input_file,
        ">",
        output_file,
    ]
    with open(input_file, "w") as out:
        for seq_id, sequence in data.items():
            out.write(f">{seq_id}\n{sequence}\n")
    sp.run(" ".join(mafft_cmd), shell=True)


def intra_cluster_heatmap(
    data: list[dict[str, str]], input_file: str, output_file: str
):
    """
    Generates heatmaps of intra-cluster sequence similarities, based off MAFFT alignments.

    Args:
        data (list[dict[str, str]]): List of clusters, where each cluster is a dictionary of sequences.
        input_file (str): Path to the input FASTA file for MAFFT.
        output_file (str): Path to the output alignment file from MAFFT.
    """

    for i, cluster in enumerate(data):
        if len(cluster) < 2:  # Skip single-sequence clusters
            continue
        run_mafft(cluster, input_file, output_file)

        similarity_df, mean_sim = get_similarity_df(output_file)
        sns.heatmap(
            similarity_df,
            cmap="viridis",
            cbar_kws={"label": "Percent Identity (%)"},
            vmin=0,
            vmax=100,
        )
        plt.show()
        print(
            f"Cluster {i} mean similarity (excluding self-similarity): {mean_sim:.2f}%"
        )


def inter_cluster_heatmap(
    data: list[dict[str, str]], input_file: str, output_file: str
):
    """
    Generates a heatmap of inter-cluster sequence similarities, based off MAFFT alignments.

    Args:
        data (list[dict[str, str]]): List of clusters, where each cluster is a dictionary of sequences.
        input_file (str): Path to the input FASTA file for MAFFT.
        output_file (str): Path to the output alignment file from MAFFT.
    """

    # Flatten dictionary
    data_flat = {}
    for cluster in data:
        for seq_id, sequence in cluster.items():
            data_flat[seq_id] = sequence
    run_mafft(data_flat, input_file, output_file)

    # Color sequences by their cluster identity
    color_dict = {}
    cluster_colors = sns.color_palette("hsv", len(data))
    for cluster_idx, cluster in enumerate(data):
        for seq_id in cluster.keys():
            color_dict[seq_id] = cluster_colors[cluster_idx]

    # Order rows in clustermap by cluster
    row_order = []
    for cluster in data:
        row_order.extend(cluster.keys())

    similarity_df, mean_sim = get_similarity_df(output_file)
    similarity_df = similarity_df.loc[row_order, row_order]
    row_colors = similarity_df.index.map(color_dict)
    sns.clustermap(
        similarity_df,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap="viridis",
        cbar_kws={"label": "Percent Identity (%)"},
        row_cluster=False,
        col_cluster=False,
        vmin=0,
        vmax=100,
    )
    plt.show()
    print(f"Inter-cluster mean similarity (excluding self-similarity): {mean_sim:.2f}%")


def get_intra_inter_similarities(
    data: list[dict[str, str]], input_file: str, output_file: str
) -> tuple[list[float], list[float]]:
    """
    Returns lists of intra- and inter-cluster pairwise similarity values for a family.

    Args:
        data (list[dict[str, str]]): List of clusters.
        input_file (str): Path to the input FASTA file for MAFFT.
        output_file (str): Path to the output alignment file from MAFFT.

    Returns:
        tuple[list[float], list[float]]: (intra_sims, inter_sims)
    """

    intra_sims, inter_sims = [], []

    # Intra-cluster: align each cluster independently
    for cluster in data:
        if len(cluster) < 2:
            continue
        run_mafft(cluster, input_file, output_file)
        similarity_df, _ = get_similarity_df(output_file)
        seq_ids = list(similarity_df.index)
        for i in range(len(seq_ids)):
            for j in range(i + 1, len(seq_ids)):
                intra_sims.append(similarity_df.iloc[i, j])

    # Inter-cluster: align each pair of clusters together
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            combined = {**data[i], **data[j]}
            run_mafft(combined, input_file, output_file)
            similarity_df, _ = get_similarity_df(output_file)
            ids_i = list(data[i].keys())
            ids_j = list(data[j].keys())
            for id_i in ids_i:
                for id_j in ids_j:
                    inter_sims.append(similarity_df.loc[id_i, id_j])

    return intra_sims, inter_sims


def rep_cluster_heatmap(data: list[dict[str, str]], input_file: str, output_file: str):
    """
    Generates a heatmap of representative sequences from each cluster, based off MAFFT alignments.

    Args:
        data (list[dict[str, str]]): List of clusters, where each cluster is a dictionary of sequences.
        input_file (str): Path to the input FASTA file for MAFFT.
        output_file (str): Path to the output alignment file from MAFFT.
    """

    # Flatten dictionary to just first sequence (representative) of each cluster
    data_flat = {}
    for cluster in data:
        first_seq_id = next(iter(cluster))
        data_flat[first_seq_id] = cluster[first_seq_id]
    run_mafft(data_flat, input_file, output_file)

    row_order = []
    for cluster in data:
        row_order.extend(cluster.keys())

    similarity_df, mean_sim = get_similarity_df(output_file)
    sns.clustermap(
        similarity_df,
        cmap="viridis",
        cbar_kws={"label": "Percent Identity (%)"},
        row_cluster=False,
        col_cluster=False,
        vmin=0,
        vmax=100,
    )
    plt.show()
    print(
        f"Representative cluster mean similarity (excluding self-similarity): {mean_sim:.2f}%"
    )


# %%
# Collect intra/inter similarity values across all families for boxplots
mmseqs_results_path = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")
mafft_input, mafft_output = (
    f"{PROJECT_ROOT}/notebooks/reports/clusters.fasta",
    f"{PROJECT_ROOT}/notebooks/reports/clusters.mfa",
)
records = []
for file in sorted(mmseqs_results_path.glob("*.fasta")):
    if "_all_seqs" not in file.name:
        continue

    # Ignore mmseqs results for zf-C2H2, Homeobox, and ZBTB families
    if file.name == "zf-C2H2.fasta.mmseqs_all_seqs.fasta":
        continue
    if file.name == "Homeobox.fasta.mmseqs_all_seqs.fasta":
        continue
    if file.name == "ZBTB.fasta.mmseqs_all_seqs.fasta":
        continue

    print(f"Processing {file.name}...")

    family = file.name.split(".fasta")[0]
    data = parse_fasta_clusters(file)

    intra_sims, inter_sims = get_intra_inter_similarities(
        data, mafft_input, mafft_output
    )
    for sim in intra_sims:
        records.append({"family": family, "type": "intra", "similarity": sim})
    for sim in inter_sims:
        records.append({"family": family, "type": "inter", "similarity": sim})

sim_df = pd.DataFrame(records)
sim_df.to_csv(
    f"{PROJECT_ROOT}/notebooks/reports/intra_inter_similarities.csv", index=False
)


# %%
# Plot similarity for individual clusters
mafft_input, mafft_output = (
    f"{PROJECT_ROOT}/notebooks/reports/clusters.fasta",
    f"{PROJECT_ROOT}/notebooks/reports/clusters.mfa",
)
mmseqs_results_path = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")

for file in mmseqs_results_path.glob("*.fasta"):
    if "_all_seqs" not in file.name:
        continue

    if "Fork_head" not in file.name:
        continue

    data = parse_fasta_clusters(file)
    intra_cluster_heatmap(data, mafft_input, mafft_output)
    inter_cluster_heatmap(data, mafft_input, mafft_output)
    rep_cluster_heatmap(data, mafft_input, mafft_output)
# %%
