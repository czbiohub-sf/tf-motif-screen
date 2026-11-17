"""
Clusters zf-C2H2 transcription factors using zinc finger recognition codes. Same output as mmseqs.

Ben Iovino  3/11/26 Biohub
"""

# %%
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from af3tfm.dataset import DATA_DIR
from af3tfm.utils import deduplicate_gene_symbols
from af3tfm.zfc2h2 import (
    extract_zf_codes,
    build_fingerprint,
    fingerprint_distance,
)

RESULTS_DIR = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")


# %%
# Load data
tf_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
zfc2h2 = tf_df[tf_df["Family"] == "zf-C2H2"].copy().reset_index(drop=True)
print(f"zf-C2H2 proteins: {len(zfc2h2)}")
doms_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv")

# Normalize DBID: extract Ensembl gene ID from compound key
doms_df["DBID"] = doms_df["DBID"].apply(
    lambda x: x.split()[1] if len(x.split()) > 1 else x
)
c2h2_doms = doms_df[doms_df["description"].str.contains("C2H2", na=False)]
annotated_zf_count = c2h2_doms.groupby("DBID").size().to_dict()


# %%
# Get recognition codes for each protein
fingerprints = {}  # DBID → fingerprint

for _, row in zfc2h2.iterrows():
    codes = extract_zf_codes(row["Sequence"])
    fingerprints[row["DBID"]] = build_fingerprint(codes)

# Proteins where the regex found at least one ZF
detected = {dbid for dbid, fp in fingerprints.items() if fp[0] > 0}
no_match = set(fingerprints) - detected
print(f"Proteins with ≥1 ZF detected: {len(detected)}")
print(f"Proteins with 0 ZF detected:  {len(no_match)}")

# Distribution of ZF counts
zf_dist = Counter(fp[0] for fp in fingerprints.values())
print("\nZF count distribution:")
for k in sorted(zf_dist):
    print(f"  {k:3d} ZFs: {zf_dist[k]} proteins")


# %%
# Cluster by fingerprint similarity
MISMATCH_PER_ZF = 1  # max avg mismatches per ZF code (2 flexible positions)
clusters = {}  # DBID → cluster_label (globally unique str)
cluster_id_counter = 0

# If no ZF detected, TF is alone in a cluster
for dbid in no_match:
    clusters[dbid] = f"clust_{cluster_id_counter:04d}"
    cluster_id_counter += 1

# Cluster each ZF-count group separately
zf_groups = {}
for dbid, fp in fingerprints.items():
    if dbid in no_match:
        continue
    n = fp[0]
    zf_groups.setdefault(n, []).append(dbid)

for n_zf, members in sorted(zf_groups.items()):
    if len(members) == 1:
        clusters[members[0]] = f"clust_{cluster_id_counter:04d}"
        cluster_id_counter += 1
        continue

    fps = [fingerprints[m] for m in members]
    threshold = MISMATCH_PER_ZF * n_zf  # 1 MM per ZF on avg

    # Build pairwise distance matrix
    n = len(members)
    dist_mat = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        d = fingerprint_distance(fps[i], fps[j])
        dist_mat[i, j] = dist_mat[j, i] = d

    if threshold == 0 or n_zf == 0:
        # Exact matching: each unique fingerprint is its own cluster
        fp_to_label = {}
        for i, (m, fp) in enumerate(zip(members, fps)):
            key = fp[1]  # the code tuple
            if key not in fp_to_label:
                fp_to_label[key] = f"clust_{cluster_id_counter:04d}"
                cluster_id_counter += 1
            clusters[m] = fp_to_label[key]
    else:
        # Hierarchical clustering with average linkage
        condensed = squareform(dist_mat, checks=False)
        if len(condensed) == 0:
            clusters[members[0]] = f"clust_{cluster_id_counter:04d}"
            cluster_id_counter += 1
            continue
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=threshold, criterion="distance")
        label_offset = cluster_id_counter
        for m, lbl in zip(members, labels):
            clusters[m] = f"clust_{label_offset + lbl:04d}"
        cluster_id_counter += label_offset + labels.max()

print(f"\nTotal clusters: {len(set(clusters.values()))}")
print(f"Total proteins assigned: {len(clusters)}")


# %%
# Select representative per cluster
#   1. Medoid: member minimizing total recognition-code distance to all others
#      in the cluster (most central by the metric the clusters were built on).
#      Within a cluster p-1/p3 are constant and p5 is unscored, so this is
#      effectively centrality over the flexible p2/p6 positions.
#   2. Most C2H2 domains annotated by InterProScan (tie-break, more curated evidence)
#   3. First alphabetically by DBID (members are identical, this is for reproducibility)
cluster_members = defaultdict(list)
for dbid, clust in clusters.items():
    cluster_members[clust].append(dbid)

seq_lookup = zfc2h2.set_index("DBID")["Sequence"].to_dict()

# Domain sequence lookup: concatenate C2H2 domain regions (sorted by start) per protein
dom_seq_lookup: dict[str, str] = {}
for dbid, full_seq in seq_lookup.items():
    protein_doms = c2h2_doms[c2h2_doms["DBID"] == dbid].sort_values("start")
    if protein_doms.empty:
        dom_seq_lookup[dbid] = full_seq
    else:
        dom_seq_lookup[dbid] = "".join(
            full_seq[int(r["start"]) - 1 : int(r["end"])]
            for _, r in protein_doms.iterrows()
        )

representatives = {}  # cluster_label → representative DBID
for clust, members in cluster_members.items():
    if len(members) == 1:
        representatives[clust] = members[0]
        continue

    def sort_key(dbid):
        medoid_dist = sum(
            fingerprint_distance(fingerprints[dbid], fingerprints[other])
            for other in members
            if other != dbid
        )
        annotated = annotated_zf_count.get(dbid, 0)
        return (medoid_dist, -annotated, dbid)

    rep = sorted(members, key=sort_key)[0]
    representatives[clust] = rep

print(f"Representatives selected: {len(representatives)}")


# %%
# Write outputs
tsv_path = RESULTS_DIR / "zf-C2H2.recog_cluster.tsv"
rows = []
for clust, members in cluster_members.items():
    rep = representatives[clust]
    for member in members:
        rows.append((rep, member))
rows.sort(key=lambda x: (x[0], x[1]))

tsv_df = pd.DataFrame(rows, columns=["representative", "member"])
tsv_df.to_csv(tsv_path, sep="\t", index=False, header=False)
print(f"  {len(tsv_df)} rows ({len(representatives)} clusters, {len(zfc2h2)} proteins)")

# Representative sequences
rep_fasta_path = RESULTS_DIR / "zf-C2H2.recog_rep_seq.fasta"
rep_records = []
for clust, rep_dbid in representatives.items():
    if rep_dbid not in dom_seq_lookup:
        continue
    row = zfc2h2[zfc2h2["DBID"] == rep_dbid].iloc[0]
    n_zf = fingerprints[rep_dbid][0]
    code_str = "|".join(fingerprints[rep_dbid][1]) if n_zf > 0 else "no_zf"
    rep_records.append(
        SeqRecord(
            Seq(dom_seq_lookup[rep_dbid]),
            id=rep_dbid,
            description=f"{row['TF_Name']} n_zf={n_zf} codes={code_str}",
        )
    )
rep_records.sort(key=lambda r: r.id)
with open(rep_fasta_path, "w") as f:
    SeqIO.write(rep_records, f, "fasta")
print(f"  {len(rep_records)} representative sequences")

# All sequences (reps + members of clusters)
all_seqs_path = RESULTS_DIR / "zf-C2H2.recog_all_seqs.fasta"
name_lookup = zfc2h2.set_index("DBID")["TF_Name"].to_dict()
with open(all_seqs_path, "w") as f:
    for clust, rep_dbid in sorted(representatives.items(), key=lambda x: x[1]):
        f.write(f">{rep_dbid}\n")  # rep ID only to match mmseqs format
        members = cluster_members[clust]  # all members, each with sequence
        rep_first = [rep_dbid] + [m for m in sorted(members) if m != rep_dbid]
        for member in rep_first:
            if member not in dom_seq_lookup:
                continue
            name = name_lookup.get(member, "")
            f.write(f">{member} {name}\n{dom_seq_lookup[member]}\n")


# %%
# Summary of clusters
sizes = [len(v) for v in cluster_members.values()]
sizes_arr = np.array(sizes)

print("\n── Cluster size statistics ──────────────────────────────────────────")
print(f"  Singleton clusters (size=1): {(sizes_arr == 1).sum()}")
print(f"  Clusters size 2–5:           {((sizes_arr >= 2) & (sizes_arr <= 5)).sum()}")
print(f"  Clusters size 6–20:          {((sizes_arr >= 6) & (sizes_arr <= 20)).sum()}")
print(f"  Clusters size >20:           {(sizes_arr > 20).sum()}")
print(f"  Largest cluster:             {sizes_arr.max()} members")
print(f"  Mean cluster size:           {sizes_arr.mean():.2f}")
print(f"  Median cluster size:         {np.median(sizes_arr):.1f}")

# Show top 10 largest clusters with their fingerprints
top_clusters = sorted(cluster_members.items(), key=lambda x: -len(x[1]))[:10]
print("\n── Top 10 largest clusters ──────────────────────────────────────────")
for clust, mems in top_clusters:
    rep = representatives[clust]
    fp = fingerprints[rep]
    rep_name = zfc2h2[zfc2h2["DBID"] == rep]["TF_Name"].values[0]
    codes = " | ".join(fp[1][:5]) + (" ..." if fp[0] > 5 else "")
    print(
        f"  {clust} ({len(mems)} members): rep={rep_name}, n_zf={fp[0]}, codes={codes}"
    )


# %%
# Update tfdb4_rep_tfs.csv with new zf-C2H2 representatives
rep_tfs_path = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
rep_df = pd.read_csv(rep_tfs_path)
rep_df_no_zfc2h2 = rep_df[
    rep_df["Family"] != "zf-C2H2"
]  # Drop rows, if existing from mmseqs

# Get full TF data for new representatives from the all-TF dataframe
new_rep_ids = list(representatives.values())
new_zfc2h2_rows = tf_df[tf_df["DBID"].isin(new_rep_ids)].copy()

# Concatenate, deduplicate gene symbols, and save
updated_df = pd.concat([rep_df_no_zfc2h2, new_zfc2h2_rows], ignore_index=True)
updated_df = deduplicate_gene_symbols(updated_df)
updated_df.to_csv(rep_tfs_path, index=False)

# %%
