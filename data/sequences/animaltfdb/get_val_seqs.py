"""
Extracts sequences from one or more TF clusters and writes them to a CSV, then pairs
each TF with a set of HOMER motif clusters and appends them to a motifs txt file.

Ben Iovino  3/24/26    CZ-Biohub
"""

# %%
import csv
import re
from pathlib import Path

from af3tfm.dataset import DATA_DIR

RESULTS_DIR = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")
TFS_CSV = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
OUTPUT_CSV = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_val_tfs.csv")
HOMER_DIR = Path(f"{DATA_DIR}/tools/homer/jaspar_clustered/homerResults")
OUTPUT_TXT = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_val_motifs.txt")


def get_cluster_members(family: str, cluster_type: str, reps: list[str]) -> list[str]:
    """Returns all member IDs belonging to the given representative sequences.

    Args:
        family (str): TF family name (e.g. 'zf-C2H2', 'Homeobox').
        cluster_type (str): Clustering type, either 'mmseqs' or 'recog'.
        reps (list[str]): Representative sequence IDs to extract clusters for.

    Returns:
        list[str]: Member sequence IDs across all requested clusters.
    """

    if cluster_type == "recog":
        tsv_path = RESULTS_DIR / f"{family}.recog_cluster.tsv"
    else:
        tsv_path = RESULTS_DIR / f"{family}.fasta.mmseqs_cluster.tsv"

    rep_set = set(reps)
    members = []
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep in rep_set:
                members.append(member)

    return members


def get_sequences(members: list[str]) -> list[dict]:
    """Looks up sequences for the given member IDs in the TFs CSV.

    Args:
        members (list[str]): Sequence IDs to look up.

    Returns:
        list[dict]: Rows from tfdb4_tfs.csv for the given member IDs.
    """

    member_set = set(members)
    rows = []
    with open(TFS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["DBID"] in member_set:
                rows.append(row)

    missing = member_set - {r["DBID"] for r in rows}
    if missing:
        print(f"Warning: {len(missing)} IDs not found: {missing}")

    return rows


def get_motif_cluster(motif_rep: str) -> list[str]:
    """Returns the representative motif and all of its similar motifs from the HOMER results.

    Args:
        motif_rep (str): Representative motif name (e.g. 'motif8').

    Returns:
        list[str]: Motif names, starting with the representative followed by similar motifs
            in numerical order.
    """

    similar_files = sorted(
        HOMER_DIR.glob(f"{motif_rep}.similar*.motif"),
        key=lambda f: int(re.search(r"similar(\d+)", f.name).group(1)),
    )
    motifs = [motif_rep] + [f.stem for f in similar_files]

    return motifs


def write_val_data(
    family: str,
    cluster_type: str,
    seq_reps: list[str],
    motif_reps: list[str],
):
    """Extracts TF sequences and motif pairings for a given family and appends both output files.

    Args:
        family (str): TF family name (e.g. 'Homeobox').
        cluster_type (str): Clustering type, either 'mmseqs' or 'recog'.
        seq_reps (list[str]): Representative sequence IDs for the mmseqs clusters.
        motif_reps (list[str]): Representative motif names (e.g. ['motif5', 'motif8']).
    """

    # Sequences
    members = get_cluster_members(family, cluster_type, seq_reps)
    rows = get_sequences(members)

    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["DBID", "TF_Name", "Family", "Sequence", "MSA_Depth"],
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} sequences to {OUTPUT_CSV.name}")

    # Motifs
    motifs = []
    for rep in motif_reps:
        motifs.extend(get_motif_cluster(rep))
    motif_str = ",".join(motifs)

    with open(OUTPUT_TXT, "a") as f:
        for row in rows:
            f.write(f"{row['DBID'].lower()},{motif_str}\n")
    print(
        f"Appended {len(rows)} TFs with {len(motifs)} motifs to {OUTPUT_TXT.name} ({', '.join(motif_reps)})"
    )


# %%
# zf-C2H2 clusters (recog)
write_val_data(
    "zf-C2H2", "recog", ["ENSDARG00000037421"], ["motif172", "motif271", "motif297"]
)
write_val_data(
    "zf-C2H2", "recog", ["ENSDARG00000015554"], ["motif172", "motif271", "motif297"]
)

# %%
# Homeobox clusters (both recog)
write_val_data("Homeobox", "recog", ["ENSDARG00000010630"], ["motif109", "motif144"])
write_val_data("Homeobox", "recog", ["ENSDARG00000042032"], ["motif5"])

# %%
# Fork_head cluster (mmseqs) - large 54-member cluster spanning the FOX family.
write_val_data("Fork_head", "mmseqs", ["ENSDARG00000093129"], ["motif86", "motif16"])

# %%
# bHLH clusters
write_val_data(
    "bHLH",
    "mmseqs",
    ["ENSDARG00000041689", "ENSDARG00000041689", "ENSDARG00000029830"],
    ["motif2", "motif49", "motif66", "motif72", "motif239", "motif259", "motif294"],
)

# %%
