"""
Here we are trying to determine the top motifs predicted for each TF based on confidence metrics
so that we can sample more heavily from these motifs in downstream predictions.
"""

# %%
from collections import defaultdict

import pandas as pd

from af3tfm.dataset import DATA_DIR, DS
from af3tfm.jobdb import Database
from af3tfm.tfactor import load_tfs
from af3tfm.utils import set_plotting_style

set_plotting_style()

# Load TF information
dataset = DS["tfdb4_rep"]
tf_df = pd.read_csv(dataset.seq_csv)
tfs = load_tfs(tf_df, name="DBID")

# Read job database
run = "hero"
db = Database(
    job_path=f"{DATA_DIR}/jobs/{run}",
    dataset=dataset.name,
)

# Query database to get ipTM scores and collect by motif
query = """
SELECT protein, dna, motif, ipTM, dom_pLDDT, dna_pLDDT
FROM runs
WHERE structure_path NOT LIKE '%output_fold_top_50%'
"""
scores = defaultdict(list)
for protein, dna, motif, iptm, dom_plddt, dna_plddt in db.cursor.execute(
    query
).fetchall():
    scores[protein].append((motif, dna, iptm, dom_plddt, dna_plddt))

db.conn.close()

# %%
top_motifs, top_n = {}, 50
for tf_name in tf_df["TF_Name"]:
    # Get scores for this TF
    dbid = tf_df[tf_df["TF_Name"] == tf_name]["DBID"].values[0].lower()
    motif_scores = scores[dbid]
    motif_scores.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)

    # Collect scores per motif
    motif_averages = defaultdict(list)
    for motif, dna, iptm, dom_plddt, dna_plddt in motif_scores:
        motif_averages[motif].append((iptm, dom_plddt))

    # Get average scores per motif
    for motif in motif_averages:
        iptm_values = [x[0] for x in motif_averages[motif]]
        plddt_values = [x[1] for x in motif_averages[motif]]
        avg_iptm = sum(iptm_values) / len(iptm_values)
        avg_plddt = sum(plddt_values) / len(plddt_values)
        motif_averages[motif] = (avg_iptm, avg_plddt)

    # Sort motifs by average ipTM and pLDDT
    sorted_motifs = sorted(
        motif_averages.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True
    )
    top_motifs[dbid] = [motif for motif, scores in sorted_motifs[:top_n]]

# %%
with open(
    f"{DATA_DIR}/sequences/{dataset.name}_{run}_top_{top_n}_motifs.txt", "w"
) as f:
    for dbid, motifs in top_motifs.items():
        if motifs is None:
            f.write(f"{dbid},None\n")
        else:
            f.write(f"{dbid},{','.join(motifs)}\n")
# %%
