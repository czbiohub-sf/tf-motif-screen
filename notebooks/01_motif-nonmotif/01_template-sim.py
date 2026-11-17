"""
Compute template sequence identity for each TF from AF3 input data.json files
and write the results as a new column in cisbp_tfs.csv.
"""

# %%
import json
from pathlib import Path

import pandas as pd
from Bio import Align

from af3tfm.dataset import DATA_DIR, DS
from af3tfm.utils import set_plotting_style

set_plotting_style()

dataset = DS["cisbp"]
output_data = Path(f"{DATA_DIR}/jobs/motif-nonmotif/output_data")

AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


def parse_template_seq(mmcif: str) -> str:
    """Extract the 1-letter sequence from an mmcif string via _entity_poly_seq."""
    in_loop, cols, seq = False, [], []
    for line in mmcif.splitlines():
        line = line.strip()
        if line == "loop_":
            in_loop = True
            cols = []
            continue
        if in_loop and line.startswith("_entity_poly_seq."):
            cols.append(line.split(".")[1])
            continue
        if in_loop and cols and not line.startswith("_"):
            if not line or line.startswith("#"):
                in_loop = False
                cols = []
                continue
            parts = line.split()
            if len(parts) >= len(cols):
                row = dict(zip(cols, parts))
                seq.append(AA3TO1.get(row.get("mon_id", ""), "X"))
    return "".join(seq)


_aligner = Align.PairwiseAligner()
_aligner.mode = "global"
_aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
_aligner.open_gap_score = -11
_aligner.extend_gap_score = -1
# Free end gaps for the template so the full DBD is always aligned
_aligner.target_left_open_gap_score = 0
_aligner.target_left_extend_gap_score = 0
_aligner.target_right_open_gap_score = 0
_aligner.target_right_extend_gap_score = 0


def template_seqid(
    data_json: Path, dbd_start: int, dbd_end: int
) -> tuple[float, str | None]:
    """
    Return the maximum DBD sequence identity and name of the best template.

    Extracts the DBD subsequence from the query and locally aligns it against
    each template sequence, taking the best identity across all templates.
    """
    with open(data_json) as f:
        d = json.load(f)

    query_seq = d["sequences"][0]["protein"]["sequence"]
    dbd_seq = query_seq[dbd_start - 1 : dbd_end]  # convert to 0-based slice
    templates = d["sequences"][0]["protein"]["templates"]

    best, best_name = 0.0, None
    for t in templates:
        tmpl_seq = parse_template_seq(t["mmcif"])
        if not tmpl_seq:
            continue
        aln = next(iter(_aligner.align(tmpl_seq, dbd_seq)), None)
        if aln is None:
            continue
        counts = aln.counts()
        aligned = counts.identities + counts.mismatches
        seqid = counts.identities / aligned if aligned > 0 else 0.0
        print(
            f"  Template {t['mmcif'].splitlines()[0].removeprefix('data_')}: {seqid:.1%} identity"
        )
        if seqid > best:
            best = seqid
            best_name = t["mmcif"].splitlines()[0].removeprefix("data_")
            best_align = aln

    return round(best * 100, 1), best_name, best_align


# %%
df = pd.read_csv(dataset.seq_csv)
doms_df = pd.read_csv(dataset.dom_csv)
seqids = []

for _, row in df.iterrows():
    if row["TF_Name"] != "prkrira":
        continue
    dbid = row["DBID"].lower()
    print(dbid)
    data_json = output_data / dbid / f"{dbid}_data.json"

    dbd = doms_df.loc[(doms_df["TF_Name"] == row["TF_Name"]) & doms_df["DNA-Binding"]]
    if dbd.empty:
        print(f"{row['TF_Name']}: no DBD found")
        seqids.append(None)
        continue

    dbd_start, dbd_end = dbd.iloc[0]["start"], dbd.iloc[0]["end"]

    if data_json.exists():
        sid, tmpl_name, best_align = template_seqid(data_json, dbd_start, dbd_end)
        print(f"{row['TF_Name']} (DBD {dbd_start}-{dbd_end}): {sid:.1f}% [{tmpl_name}]")
        print(best_align)
    else:
        sid = None
        print(f"{row['TF_Name']}: no data.json found")
    seqids.append(sid)

df["Template_PID"] = seqids
df.to_csv(dataset.seq_csv, index=False)
print(f"\nWrote Template_PID to {dataset.seq_csv}")
# %%
