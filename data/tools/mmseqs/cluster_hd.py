"""
Clusters homeodomain transcription factors by recognition code extracted from
a hmmalign alignment to PF00046 (Homeobox domain HMM). Proteins are first
classified by co-occurring secondary domains (POU, LIM-HD, PAX, etc.) and
then clustered within each class.

Ben Iovino  3/26/26 Biohub
"""

# %%
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path
import urllib.request
import pandas as pd
from af3tfm.dataset import DATA_DIR
from af3tfm.homeobox import (
    HD_POSITIONS,
    CANONICAL_P51,
    CANONICAL_P51_MS,
    IPR_ACCESSION,
    HMM_PATH,
    load_hmm,
    align_sequences,
    extract_recognition_codes,
    recognition_code_str,
    extract_hd_sequences,
)
from af3tfm.utils import deduplicate_gene_symbols

TFS_PATH = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
DOMS_PATH = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv")
OUT_DIR = Path(f"{DATA_DIR}/tools/mmseqs/hd_clusters")
# HD class definitions
HD_CLASS_DOMAINS: list[tuple[str, set[str]]] = [
    ("POU", {"IPR000327"}),  # POU-specific domain; composite POU-S+HD site
    ("PAX", {"IPR001523"}),  # Paired domain; primary DNA contact
    ("LIM-HD", {"IPR001781"}),  # LIM zinc fingers; cofactor-dependent binding
    ("CUT-HD", {"IPR003350"}),  # CUT domain; independent DNA-binding activity
    ("SIX", {"IPR031701"}),  # SD domain; combined SD+HD MEF3-site recognition
    ("HNF1", {"IPR006899"}),  # HNF1 dimerisation domain; dimeric binding mode
    ("ADNP", {"IPR045762"}),  # ADNP zinc finger; atypical HD context
]


# %%
def download_hmm(hmm_path: Path) -> None:
    """
    Download the PF00046 HMM file from the Pfam API.

    Args:
        hmm_path: Path to save the HMM file (can be gzipped or plain text)
    """

    try:
        url = "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/PF00046/?annotation=hmm"
        urllib.request.urlretrieve(url, hmm_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download PF00046 HMM: {e}\n"
            "Manual fallback: https://pfam.xfam.org/family/PF00046/hmm"
        ) from e


def classify_hd_proteins(doms_df: pd.DataFrame) -> dict[str, str]:
    """
    Assign each homeodomain-containing protein an hd_class based on
    co-occurring secondary domains, using the priority order in HD_CLASS_DOMAINS.

    Args:
        doms_df: DataFrame containing domain annotations with columns 'DBID' and 'accession'

    Returns:
        dict mapping protein_id → hd_class  (e.g. "POU", "LIM-HD", "SOLO")
    """

    protein_accs = doms_df.groupby("DBID")["accession"].apply(set).to_dict()
    hd_proteins = set(doms_df[doms_df["accession"] == IPR_ACCESSION]["DBID"])

    result: dict[str, str] = {}
    for pid in hd_proteins:
        accs = protein_accs.get(pid, set())
        hd_class = "SOLO"
        for cls_name, cls_domains in HD_CLASS_DOMAINS:
            if accs & cls_domains:
                hd_class = cls_name
                break
        result[pid] = hd_class

    return result


def is_atypical(rec: dict[int, str]) -> bool:
    """
    Flag sequences lacking canonical Asn51 as non-DNA-binding (hopx-like).

    Args:
        rec: dict mapping match_state_position → amino_acid

    Returns:
        True if the sequence is atypical (non-DNA-binding), False otherwise
    """

    return rec.get(CANONICAL_P51_MS, "-") != CANONICAL_P51


def codes_compatible(a: dict[int, str], b: dict[int, str]) -> bool:
    """
    Two recognition codes are compatible for co-clustering if every canonical
    position (p47, p50, p51, p52, p54) is gap-free and identical in both sequences.

    Args:
        a, b: dict mapping match_state_position → amino_acid for two sequences

    Returns:
        True if the sequences are compatible, False otherwise
    """

    for pos in HD_POSITIONS:
        a_aa, b_aa = a.get(pos, "-"), b.get(pos, "-")
        if a_aa in ("-", ".") or b_aa in ("-", ".") or a_aa != b_aa:
            return False

    return True


def select_representative(
    members: list[str],
    aligned_seqs: dict[str, str],
    match_cols: list[int],
) -> str:
    """
    Select the medoid of a cluster: the member with the highest mean pairwise
    percent identity to all other members, computed across all HMM match-state
    columns. Gaps in either sequence are excluded from both num and denom.

    Args:
        members: list of instance_ids in the cluster
        aligned_seqs: dict mapping instance_id → aligned sequence string
        match_cols: list of 0-based column indices for HMM match states

    Returns:
        instance_id of the representative sequence for the cluster
    """

    if len(members) == 1:
        return members[0]

    cols: dict[str, list[str]] = {
        m: [
            aligned_seqs[m][col] if col < len(aligned_seqs[m]) else "-"
            for col in match_cols
        ]
        for m in members
    }

    mean_pid: dict[str, float] = {}
    for m in members:
        total = 0.0
        for other in members:
            if other == m:
                continue
            matches = ungapped = 0
            for a, b in zip(cols[m], cols[other]):
                if a in ("-", ".") or b in ("-", "."):
                    continue
                ungapped += 1
                if a == b:
                    matches += 1
            total += matches / ungapped if ungapped > 0 else 0.0
        mean_pid[m] = total / (len(members) - 1)

    return max(members, key=lambda m: mean_pid[m])


def cluster_sequences(
    rec_codes: dict[str, dict[int, str]],
    aligned_seqs: dict[str, str],
    match_cols: list[int],
) -> tuple[dict[str, int], list[str]]:
    """
    Two-pass clustering by recognition code compatibility with medoid representative
    selection.

    Pass 1 — group sequences by recognition code compatibility (greedy).
    Pass 2 — for each group, select the member with the highest mean pairwise
              percent identity across all HMM match-state columns as representative.

    Args:
        rec_codes: dict mapping instance_id → {match_state_position: amino_acid}
        aligned_seqs: dict mapping instance_id → aligned sequence string
        match_cols: list of 0-based column indices for HMM match states

    Returns:
        assignments    : instance_id → cluster_id  (-1 for atypicals)
        representatives: list of representative instance_ids, indexed by cluster_id
    """

    groups: list[list[str]] = []
    group_recs: list[dict[int, str]] = []
    assignments: dict[str, int] = {}

    for name, rec in rec_codes.items():
        if is_atypical(rec):
            assignments[name] = -1
            continue
        for gid, grp_rec in enumerate(group_recs):
            if codes_compatible(rec, grp_rec):
                groups[gid].append(name)
                assignments[name] = gid
                break
        else:
            assignments[name] = len(groups)
            groups.append([name])
            group_recs.append(rec)

    rep_ids: list[str] = [
        select_representative(members, aligned_seqs, match_cols) for members in groups
    ]

    return assignments, rep_ids


def build_output_df(
    all_assignments: dict[str, dict[str, int]],
    all_representatives: dict[str, list[str]],
    rec_codes: dict[str, dict[int, str]],
) -> pd.DataFrame:
    """
    Build a DataFrame summarizing the clustering results with columns:
        - protein: base protein ID without __hd suffix
        - hd_instance: full instance ID with __hd suffix for multi-HD proteins
        - hd_class: secondary domain context class (e.g. "POU", "LIM-HD", "SOLO")
        - cluster_id: integer cluster ID (starting from 0), or -1 for atypical sequences
        - representative: instance_id of the cluster representative, or None for atypical sequences
        - is_representative: True if this instance is the representative for its cluster, False otherwise
        - is_atypical: True if this instance is classified as atypical (non-DNA-binding), False otherwise
        - recognition_code: compact string representation of the recognition code (e.g. "N-S-Q-K-A" or "no_zf")
    """
    rows = []
    for hd_class, assignments in all_assignments.items():
        representatives = all_representatives[hd_class]
        for instance_id, cid in assignments.items():
            protein = instance_id.split("__hd")[0]
            rec = rec_codes[instance_id]
            rows.append(
                {
                    "protein": protein,
                    "hd_instance": instance_id,
                    "hd_class": hd_class,
                    "cluster_id": cid,
                    "representative": representatives[cid].split("__hd")[0]
                    if cid >= 0
                    else None,
                    "is_representative": (
                        cid >= 0 and instance_id == representatives[cid]
                    ),
                    "is_atypical": cid == -1,
                    "recognition_code": recognition_code_str(rec),
                    "p47": rec.get(46, "-"),
                    "p50": rec.get(49, "-"),
                    "p51": rec.get(50, "-"),
                    "p52": rec.get(51, "-"),
                    "p54": rec.get(53, "-"),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["hd_class", "cluster_id", "is_representative"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    return df


# %%
# Load TF and domain data
tfs_df = pd.read_csv(TFS_PATH)
doms_df = pd.read_csv(DOMS_PATH)
for col in doms_df.select_dtypes(include="object").columns:
    doms_df[col] = doms_df[col].str.strip()

# Extract the gene_id (2nd field) so it matches tfs_df["DBID"].
if doms_df["DBID"].str.contains("\t", na=False).any():
    doms_df["DBID"] = doms_df["DBID"].str.split("\t").str[1]


# %%
# Extract homeodomain sequences, load HMM, and align sequences
hd_seqs = extract_hd_sequences(tfs_df, doms_df)
OUT_DIR.mkdir(parents=True, exist_ok=True)
if not HMM_PATH.exists():
    download_hmm(HMM_PATH)
hmm = load_hmm(HMM_PATH)
aligned_seqs, match_cols = align_sequences(hmm, hd_seqs)


# %%
# Extract recognition codes and classify by secondary domain context
rec_codes = extract_recognition_codes(aligned_seqs, match_cols)
protein_classes = classify_hd_proteins(doms_df)
instance_classes = {
    iid: protein_classes.get(iid.split("__hd")[0], "SOLO") for iid in rec_codes
}
for cls in sorted(set(instance_classes.values())):
    n = sum(1 for v in instance_classes.values() if v == cls)
    print(f"  {cls:8s}: {n} instance(s)")


# %%
# Cluster within each HD class
all_assignments: dict[str, dict[str, int]] = {}
all_representatives: dict[str, list[str]] = {}

for hd_class in sorted(set(instance_classes.values())):
    class_recs = {
        iid: rec for iid, rec in rec_codes.items() if instance_classes[iid] == hd_class
    }
    assignments, representatives = cluster_sequences(
        class_recs, aligned_seqs, match_cols
    )
    all_assignments[hd_class] = assignments
    all_representatives[hd_class] = representatives

    n_canonical = sum(1 for v in assignments.values() if v >= 0)
    n_atypical = sum(1 for v in assignments.values() if v < 0)
    n_clusters = len(representatives)
    print(
        f"  {hd_class:8s}: {n_canonical} canonical → {n_clusters} clusters, "
        f"{n_atypical} atypical"
    )


# %%
# Build output and print summary
df = build_output_df(all_assignments, all_representatives, rec_codes)

canonical = df[~df["is_atypical"]]
print("\nTop recognition codes per class:")
for hd_class, grp in canonical.groupby("hd_class"):
    print(f"\n  [{hd_class}]")
    for code, n in grp["recognition_code"].value_counts().head(5).items():
        print(f"    {code}  {n}")

if df["is_atypical"].any():
    print("\nAtypical sequences (cluster_id = -1):")
    print(
        df[df["is_atypical"]][["protein", "hd_class", "recognition_code"]].to_string(
            index=False
        )
    )


# %%
# Write mmseqs-format outputs: cluster TSV, rep FASTA, all-seqs FASTA
RESULTS_DIR = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")
seq_lookup = tfs_df.set_index("DBID")["Sequence"].to_dict()
name_lookup = tfs_df.set_index("DBID")["TF_Name"].to_dict()

# Domain sequence lookup: protein_id → homeodomain sequence (first instance for multi-HD proteins)
hd_dom_seq_lookup: dict[str, str] = {}
for instance_id, dom_seq in hd_seqs.items():
    protein_id = instance_id.split("__hd")[0]
    if protein_id not in hd_dom_seq_lookup:
        hd_dom_seq_lookup[protein_id] = dom_seq

# Build (representative, member) pairs at the protein level, deduped per cluster
canonical = df[~df["is_atypical"]]
tsv_rows = (
    canonical[["representative", "protein"]]
    .drop_duplicates()
    .sort_values(["representative", "protein"])
)

tsv_path = RESULTS_DIR / "Homeobox.recog_cluster.tsv"
tsv_rows.to_csv(tsv_path, sep="\t", index=False, header=False)
print(
    f"  {len(tsv_rows)} rows ({canonical['representative'].nunique()} clusters, "
    f"{canonical['protein'].nunique()} proteins)"
)

# Representative sequences FASTA
rep_records = []
for _, row in (
    canonical[canonical["is_representative"]]
    .drop_duplicates("representative")
    .iterrows()
):
    rep = row["representative"]
    if rep not in hd_dom_seq_lookup:
        continue
    rep_records.append(
        SeqRecord(
            Seq(hd_dom_seq_lookup[rep]),
            id=rep,
            description=f"{name_lookup.get(rep, '')} hd_class={row['hd_class']} "
            f"recog={row['recognition_code']}",
        )
    )
rep_records.sort(key=lambda r: r.id)
rep_fasta_path = RESULTS_DIR / "Homeobox.recog_rep_seq.fasta"
with open(rep_fasta_path, "w") as f:
    SeqIO.write(rep_records, f, "fasta")
print(f"  {len(rep_records)} representative sequences → {rep_fasta_path}")

# All-sequences FASTA (mmseqs format: bare >rep_id header, then members with sequences)
all_seqs_path = RESULTS_DIR / "Homeobox.recog_all_seqs.fasta"
with open(all_seqs_path, "w") as f:
    for rep, grp in canonical.drop_duplicates("protein").groupby("representative"):
        f.write(f">{rep}\n")
        proteins = [rep] + [p for p in sorted(grp["protein"].unique()) if p != rep]
        for protein in proteins:
            if protein not in hd_dom_seq_lookup:
                continue
            f.write(
                f">{protein} {name_lookup.get(protein, '')}\n{hd_dom_seq_lookup[protein]}\n"
            )


# %%
# Update tfdb4_rep_tfs.csv: drop existing Homeobox reps, add new ones
rep_tfs_path = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
rep_df = pd.read_csv(rep_tfs_path)

rep_df_no_hd = rep_df[rep_df["Family"] != "Homeobox"]

new_rep_ids = canonical[canonical["is_representative"]]["protein"].unique()
new_hd_rows = tfs_df[tfs_df["DBID"].isin(new_rep_ids)].copy()

updated_df = pd.concat([rep_df_no_hd, new_hd_rows], ignore_index=True)
updated_df = updated_df.drop_duplicates(subset=["DBID"])
updated_df = deduplicate_gene_symbols(updated_df)
updated_df.to_csv(rep_tfs_path, index=False)

# %%
# Add in SELEX TFs so we predict their structures as well
all_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
rep_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
selex_df = pd.read_csv(f"{DATA_DIR}/sequences/selex/selex_tfs.csv")

fam_lookup = all_df.set_index("DBID")["Family"]
selex_df["Family"] = selex_df["DBID"].str.split("_").str[0].map(fam_lookup)

rep_df = rep_df[~rep_df["DBID"].isin(selex_df["DBID"])]
combined_df = pd.concat([rep_df, selex_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=["DBID"])
combined_df.to_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv", index=False)

# %%
