"""
Clusters transcription factor sequences from AnimalTFDB4 based on their domain sequences. Add in
SELEX TFs to ensure we predict structures for them as well.

Ben Iovino  12/8/25    CZ-Biohub
"""

# %%
import json
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path
from af3tfm.dataset import DATA_DIR
from af3tfm.utils import deduplicate_gene_symbols
import subprocess as sp


# %%
def parse_results(results_file_path: str, doms_df_file_path: str):
    """
    Parses results file from interproscan and writes the domains to a csv file.

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
        dbid = result["xref"][0]["name"]
        for match in result["matches"]:
            if match["signature"]["signatureLibraryRelease"]["library"] in databases:
                try:
                    for location in match[
                        "locations"
                    ]:  # Multiple locations for a single domain
                        domains.append(
                            {
                                "DBID": dbid,
                                "library": match["signature"][
                                    "signatureLibraryRelease"
                                ]["library"],
                                "accession": match["signature"]["entry"]["accession"],
                                "description": match["signature"]["entry"][
                                    "description"
                                ],
                                "start": location["start"],
                                "end": location["end"],
                            }
                        )
                except TypeError:  # No result?
                    continue

    df = pd.DataFrame(domains).set_index("DBID")
    df.to_csv(doms_df_file_path)


parse_results(
    f"{DATA_DIR}/tools/ipscan/tfdb4/tfdb4_results.json",
    f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv",
)


# %%
# Get matching Gene IDs for domain and TF dataframes
doms_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv")
for dbid in doms_df["DBID"]:
    if len(dbid.split()) > 1:
        new_dbid = dbid.split()[1]  # gene ID
        doms_df.loc[doms_df["DBID"] == dbid, "DBID"] = new_dbid

tf_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
tf_df["Domain_Sequence"] = ""

# Subset sequence for each domain
for i, row in tf_df.iterrows():
    seq = row.Sequence
    new_seq = ""
    for j, dom_row in doms_df[doms_df["DBID"] == row["DBID"]].iterrows():
        new_seq += seq[dom_row.start : dom_row.end]
    tf_df.at[i, "Domain_Sequence"] = new_seq


# %%
# For each family, write sequences to fasta file
fams_path = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/families")
fams_path.mkdir(exist_ok=True)
for fam, group in tf_df.groupby("Family"):
    with open(f"{fams_path}/{fam}.fasta", "w") as f:
        records = [
            SeqRecord(
                Seq(row["Domain_Sequence"]), id=row["DBID"], description=row["TF_Name"]
            )
            for _, row in group.iterrows()
        ]
        SeqIO.write(records, f, "fasta")


# %%
# Run mmseqs on each fasta file in fams_path
results_path = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")
results_path.mkdir(exist_ok=True)
for fasta_file in fams_path.glob("*.fasta"):
    print(f"Clustering {fasta_file.name}...")
    output_file = f"{results_path}/{fasta_file.name}.mmseqs"
    sp.run(
        ["mmseqs", "easy-cluster", str(fasta_file), str(output_file), "tmp"], check=True
    )


# %%
# Collect representative sequences from each family
results_path = Path(f"{DATA_DIR}/tools/mmseqs/animaltfdb4/results")
seq_ids = []
for results_file in results_path.glob("*_rep_seq.fasta"):
    records = list(SeqIO.parse(results_file, "fasta"))
    seq_ids.extend([record.id for record in records])
print(len(seq_ids))

# Subset dataframe for rep TFs
tf_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
rep_df = tf_df[tf_df["DBID"].isin(seq_ids)]
rep_df = rep_df[
    ~rep_df["Family"].isin(["zf-C2H2", "ZBTB", "Homeobox"])
]  # clustered later
rep_df = deduplicate_gene_symbols(rep_df)
rep_df.to_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv", index=False)


# %%
