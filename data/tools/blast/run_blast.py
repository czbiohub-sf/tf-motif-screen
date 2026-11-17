"""
Run BLASTP for each TF sequence in the dataset against all human proteins.
"""

# %%
import pandas as pd
from pathlib import Path
import subprocess as sp
from af3tfm.dataset import DS


dataset = DS["tfdb4_rep"]
tf_df = pd.read_csv(dataset.seq_csv, sep=",", header=0)
doms_df = pd.read_csv(dataset.dom_csv, sep=",", header=0)

fasta_path = Path("fasta_files")
results_path = Path("results")
fasta_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

# %%
# Write each TF to a fasta file (keyed by DBID, which is unique; TF_Name is not)
for i, row in tf_df.iterrows():
    dbid = row["DBID"]
    tf_seq = row["Sequence"]
    tf_fasta_file_path = Path(f"{fasta_path}/{dbid}.fasta")
    tf_fasta_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tf_fasta_file_path, "w") as fasta_file:
        fasta_file.write(f">{dbid}\n")
        fasta_file.write(f"{tf_seq}\n")

    # If results file already exists, skip
    blast_result_file = results_path / f"{dbid}_results.txt"
    if blast_result_file.exists():
        print(f"Results for {dbid} already exist, skipping BLAST.")
        continue

    # Call blast search
    tool_command = f"blastp -query {tf_fasta_file_path} -db humandb/humandb -out {blast_result_file} -evalue 1e-5"
    full_command = f"module load blast && {tool_command}"
    print(f"Running command: {full_command}")
    sp.run(full_command, shell=True)

# %%
